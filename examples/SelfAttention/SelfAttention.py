import sys
sys.path.append("/home/caohanghang/gitproject/Telechat/download/TeleChat-12B")
import torch
import os
from torch import nn
from torch.nn import functional as F
import math
from typing import Tuple
""" Telechat configuration"""

from packaging import version
from collections import OrderedDict
from transformers.utils import is_torch_available, logging
from transformers.configuration_utils import PretrainedConfig
from typing import TYPE_CHECKING, Any, List, Mapping, Optional


try:
    from einops import rearrange
except ImportError:
    rearrange = None

logger = logging.get_logger(__name__)

dtype = torch.float16

class TelechatConfig(PretrainedConfig):
    """
    Args:
        vocab_size (`int`, *optional*, defaults to 160256): Vocabulary size of the Telechat model.
        hidden_size (`int`, *optional*, defaults to 4096): Dimensionality of the embeddings and hidden states.
        ffn_hidden_size (`int`, *optional*, defaults to 12288): Dimensionality of the feed-forward hidden states.
        n_layer (`int`, *optional*, defaults to 30): Number of hidden layers in the Transformer
        n_head (`int`, *optional*, defaults to 32): Number of attention heads for each attention layer.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5): The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02): The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`): If enabled, use the layer norm of the hidden states as the residual in the transformer blocks
        hidden_dropout (`float`, *optional*, defaults to 0.0): Dropout rate of the dropout function on the bias dropout.
        attention_dropout (`float`, *optional*, defaults to 0.0): Dropout rate applied to the attention probs
        use_cache (`bool`, *optional*, defaults to `True`): Whether or not the model should return the last key/values attentions.
        training_seqlen (`int`, *optional*, defaults to 8192): Sequence length during last finetuning.
        logn (`bool`, *optional*, defaults to `True`): Whether or not to use logN during extrapolation.
        embed_layernorm (`bool`, *optional*, defaults to `True`): Whether or not to use embedding layernorm.

    """

    model_type = "telechat"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }

    def __init__(
        self,
        vocab_size=120000,
        hidden_size=5120,
        n_layer=38,
        n_head=32,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        ffn_hidden_size=12288,
        training_seqlen = 8192,
        logn = True,
        embed_layernorm = False,
        **kwargs,
    ):
        self.flash_attn = False
        self.vocab_size = vocab_size
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.logn = logn
        self.ffn_hidden_size = ffn_hidden_size
        self.training_seqlen = training_seqlen
        self.embed_layernorm = embed_layernorm


        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)



def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


def apply_rotary_pos_emb_torch(q, k, cos, sin, offset: int = 0):  # jitting fails with bf16
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RotaryEmbedding(torch.nn.Module):
    # Extracted from: https://github.com/EleutherAI/gpt-neox
    def __init__(self, dim, config, base=10000):
        super().__init__()
        self.config = config
        self.dim = dim
        self.base = base
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def get_mscale(self, scale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def get_ntk_alpha(self, true_seq_len):
        context_value = math.log(true_seq_len / 4096, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
        return ntk_alpha

    def forward(self, x, dtype, seq_dim=0):
        seq_len = x.shape[seq_dim]
        self.mscale = 1.0
        if not self.training:
            seq_len = max(seq_len, self.config.training_seqlen)
            self.mscale = float(self.get_mscale(seq_len / self.config.training_seqlen))
        ntk_alpha = self.get_ntk_alpha(seq_len)
        base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
        self.inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        # if self.precision == torch.bfloat16:
        emb = emb.float() if dtype == torch.bfloat16 else emb
        # [sx, 1 (b * np), hn]
        self.cos_cached = self.mscale * emb.cos()[:, None, :].to(dtype)
        self.sin_cached = self.mscale * emb.sin()[:, None, :].to(dtype)
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

class TelechatAttention(nn.Module):
    def __init__(self, config: TelechatConfig, layer_idx):
        super().__init__()
        self.kv_cache = None
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.num_key_value_heads = self.num_heads
        kv_projection_size = self.head_dim * self.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype = torch.float16)
        self.key_value = nn.Linear(self.hidden_size, kv_projection_size * 2, bias=False, dtype = torch.float16)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, dtype = torch.float16)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config=config)
        # self.rotary_emb = torch.compile(self.rotary_emb, backend=my_backend)

        self.last_key_layer = None

    def repeat_kv(self, hidden_states, n_rep):
        slen, batch, num_key_value_heads_per_partition, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(slen, batch, num_key_value_heads_per_partition, n_rep,
                                                               head_dim)
        return hidden_states.reshape(slen, batch, num_key_value_heads_per_partition * n_rep, head_dim)

    def split_tensor_along_last_dim(self,
                                    tensor: torch.Tensor,
                                    num_partitions: int,
                                    contiguous_split_chunks: bool = False,
                                    ):

        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            residual: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
        print("hidden_states.shape:", hidden_states.shape)
        print("residual.shape: ", residual.shape)
        print("attention_mask.shape: ", attention_mask.shape)
        if layer_past is not None:
            for idx, t in enumerate(layer_past):
                print("layer_past["+str(idx) +"].shape = ", t.shape)                
            
        hidden_states = hidden_states.transpose(1, 0)
        query_layer = self.query(hidden_states)
        new_tensor_shape = query_layer.size()[:-1] + \
                           (self.num_heads,
                            self.head_dim)
        query_layer = query_layer.view(*new_tensor_shape)

        mixed_kv_layer = self.key_value(hidden_states)
        new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                           (self.num_key_value_heads,
                            2 * self.head_dim)
        mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)
        (key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_kv_layer, 2)

        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        apply_rotary_fn = apply_rotary_pos_emb_torch

        seq_len = key_layer.shape[0]
        offset = 0

        if use_cache and layer_past != None:
            past_key, past_value = layer_past
            offset = past_key.shape[0]
            seq_len += offset

        cos, sin = self.rotary_emb(value_layer, dtype=value_layer.dtype)

        query_layer, key_layer = apply_rotary_fn(query_layer, key_layer, cos, sin, offset=offset)
        if use_cache:
            if layer_past != None:
                past_key, past_value = layer_past
                key_layer = torch.cat((past_key, key_layer[-1, ...].unsqueeze(0)), dim=0)
                value_layer = torch.cat((past_value, value_layer[-1, ...].unsqueeze(0)), dim=0)
            layer_past = key_layer, value_layer
        s, bz, head, dim = value_layer.shape
        s_key = key_layer.shape[0]
        s_query = query_layer.shape[0]
        query_layer = query_layer.reshape((s_query, bz, head, dim))
        key_layer = key_layer.reshape((s_key, bz, head, dim))

        if self.config.flash_attn:
            q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous() for x in
                       (query_layer, key_layer, value_layer)]
            context_layer = self.core_attention_flash(q, k, v)
            context_layer = rearrange(context_layer, 'b s h d -> b s (h d)').contiguous()
        else:
            ##[sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.reshape(s_query, bz * self.num_heads, dim)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.reshape(s_key, bz * self.num_heads, dim)
            matmul_result = self.inv_norm_factor * torch.einsum('bik,bkj->bij', query_layer.transpose(0, 1),
                                                                key_layer.transpose(0, 1).transpose(1, 2))

            attention_scores = matmul_result.view(bz, self.num_heads, s_query, s_key)

            input_dtype = attention_scores.dtype
            if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                attention_scores = attention_scores.to(torch.float)
            attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
            attention_probs = F.softmax(attn_weights, dim=-1).to(input_dtype)  ##dtype = torch.float32
            attention_probs = self.attention_dropout(attention_probs)
            attention_probs_reshaped = attention_probs.view(bz * self.num_heads, s_query, s_key)

            value_layer = value_layer.reshape(s_key, bz * self.num_heads, dim)
            context_layer = torch.bmm(attention_probs_reshaped, value_layer.transpose(0, 1))
            context_layer = self._merge_heads(context_layer)

        output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        present = None
        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return output_tensor, layer_past


def dump_model_bin(model: TelechatAttention):
    # 随机初始化权重
    torch.save(model.state_dict(), "attention_model.bin")

def calc_model(model: TelechatAttention):
    if not os.path.exists("attention_model.bin"):
        dump_model_bin(model)
    model.load_state_dict(torch.load("attention_model.bin"))
    hidden_states = torch.ones([1, 5, 5120], dtype=torch.float16)
    residual =  torch.ones([1, 5, 5120], dtype=torch.float16)
    attention_mask = torch.ones([1, 1, 5, 5], dtype=torch.bool)
    print("output:", model(hidden_states, residual, attention_mask))


if __name__ == "__main__":
    config = TelechatConfig()
    model = TelechatAttention(config, 1)
    # 创建模型实例
    model.eval()
    calc_model(model)
