import torch
import math


class RotaryEmbedding(torch.nn.Module):
    # Extracted from: https://github.com/EleutherAI/gpt-neox
    def __init__(self, dim, base=10000):
        super().__init__()
        # self.config = config

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
            self_config_training_seqlen = 8192
            seq_len = max(seq_len, self_config_training_seqlen)
            self.mscale = float(self.get_mscale(seq_len / self_config_training_seqlen))
        ntk_alpha = self.get_ntk_alpha(seq_len)
        base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
        self.inv_freq = 1.0 / (
            base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim)
        )
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        # if self.precision == torch.bfloat16:
        emb = emb.float() if dtype == torch.bfloat16 else emb
        # [sx, 1 (b * np), hn]
        self.cos_cached = self.mscale * emb.cos()[:, None, :].to(dtype)
        self.sin_cached = self.mscale * emb.sin()[:, None, :].to(dtype)
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


if __name__ == "__main__":
    hidden_size = 5120
    n_head = 32
    head_dim = hidden_size // n_head
    rotary = RotaryEmbedding(dim=head_dim)
    rotary.eval()
    cos, sin = rotary(torch.ones(40), torch.float16)
    cos1, sin1 = rotary(torch.ones(8192), torch.float16)
    cos_equal = torch.equal(cos, cos1)
    sin_equal = torch.equal(sin, sin1)
    print("cos equal:", cos_equal)
    print("sin equal:", sin_equal)

    print("RotaryEmbedding passed")
