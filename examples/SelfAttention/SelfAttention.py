import torch
from torch import nn
import math


class TelechatAttention(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.kv_cache = None
        self.layer_idx = layer_idx

        self.hidden_size = 5120
        self.num_heads = 32
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = 0.0

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
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key_value = nn.Linear(self.hidden_size, kv_projection_size * 2, bias=False)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(self.attention_dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config=config)

        # self.core_attention_flash = FlashSelfAttention(
        #     causal=True, attention_dropout=config.attention_dropout
        # )

        self.last_key_layer = None

    def repeat_kv(self, hidden_states, n_rep):
        slen, batch, num_key_value_heads_per_partition, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(
            slen, batch, num_key_value_heads_per_partition, n_rep, head_dim
        )
        return hidden_states.reshape(
            slen, batch, num_key_value_heads_per_partition * n_rep, head_dim
        )

    def split_tensor_along_last_dim(
        self,
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
        hidden_states = hidden_states.transpose(1, 0)  # transpose

        # 生成Q矩阵，并根据multi-head 调整shape
        query_layer = self.query(hidden_states)
        new_tensor_shape = query_layer.size()[:-1] + (self.num_heads, self.head_dim)
        query_layer = query_layer.view(*new_tensor_shape)

        # 生成K, V矩阵，并根据multi-head，调整shape
        mixed_kv_layer = self.key_value(hidden_states)
        new_tensor_shape = mixed_kv_layer.size()[:-1] + (
            self.num_key_value_heads,
            2 * self.head_dim,
        )
        mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)
        (key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_kv_layer, 2)

        # 重新调整K, Q的张量形状
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        seq_len = key_layer.shape[0]
        offset = 0

        if use_cache and layer_past != None:
            past_key, past_value = layer_past
            offset = past_key.shape[0]
            seq_len += offset

        # 将旋转位置编码（RoPE）应用于 K 和 Q。
        cos, sin = self.rotary_emb(value_layer, dtype=value_layer.dtype)
        apply_rotary_fn = apply_rotary_pos_emb_torch
        query_layer, key_layer = apply_rotary_fn(
            query_layer, key_layer, cos, sin, offset=offset
        )
        # print("use_cache", use_cache)
        if use_cache:
            if layer_past != None:
                past_key, past_value = layer_past
                key_layer = torch.cat(
                    (past_key, key_layer[-1, ...].unsqueeze(0)), dim=0
                )
                value_layer = torch.cat(
                    (past_value, value_layer[-1, ...].unsqueeze(0)), dim=0
                )
            layer_past = key_layer, value_layer

        # 将 Q 和 K 的张量形状调整为标准的多头注意力机制所需的格式。
        # 提取 V 的形状信息
        s, bz, head, dim = value_layer.shape
        # 提取 K 和 Q 的序列长度
        s_key = key_layer.shape[0]
        s_query = query_layer.shape[0]
        # 重塑 K 和 Q 的形状
        query_layer = query_layer.reshape((s_query, bz, head, dim))
        key_layer = key_layer.reshape((s_key, bz, head, dim))

        # print("self.config.flash_attn: ", self.config.flash_attn)
        if self.config.flash_attn:
            q, k, v = [
                rearrange(x, "s b ... -> b s ...").contiguous()
                for x in (query_layer, key_layer, value_layer)
            ]
            context_layer = self.core_attention_flash(q, k, v)
            context_layer = rearrange(
                context_layer, "b s h d -> b s (h d)"
            ).contiguous()
        else:
            ##[sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.reshape(s_query, bz * self.num_heads, dim)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.reshape(s_key, bz * self.num_heads, dim)
            # Query 与 Key 的转置进行矩阵乘法
            matmul_result = self.inv_norm_factor * torch.einsum(
                "bik,bkj->bij",
                query_layer.transpose(0, 1),
                key_layer.transpose(0, 1).transpose(1, 2),
            )

            # 对 Q * K的结果进行mask
            attention_scores = matmul_result.view(bz, self.num_heads, s_query, s_key)
            input_dtype = attention_scores.dtype
            if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                attention_scores = attention_scores.to(torch.float)
            attn_weights = torch.masked_fill(
                attention_scores,
                attention_mask,
                torch.finfo(attention_scores.dtype).min,
            )

            # 计算权重
            attention_probs = F.softmax(attn_weights, dim=-1).to(
                input_dtype
            )  ##dtype = torch.float32
            attention_probs = self.attention_dropout(attention_probs)
            attention_probs_reshaped = attention_probs.view(
                bz * self.num_heads, s_query, s_key
            )

            value_layer = value_layer.reshape(s_key, bz * self.num_heads, dim)
            # 对V进行加权
            context_layer = torch.bmm(
                attention_probs_reshaped, value_layer.transpose(0, 1)
            )
            # 将多头merge起来
            context_layer = self._merge_heads(context_layer)

        # attention的输出经过线性层
        output_tensor = self.dense(context_layer)

        # 先进行dropout，然后再加上residual
        output_tensor = dropout_add(
            output_tensor, residual, self.hidden_dropout, self.training
        )
        present = None
        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return output_tensor, layer_past
