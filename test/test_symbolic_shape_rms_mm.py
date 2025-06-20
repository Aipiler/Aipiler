import torch
from torch import nn
from torch.export import export, Dim


class Qwen2RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, hidden_states, weight):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
        return weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class RMSNorm_Matmul(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.rmsnorm = Qwen2RMSNorm(eps)

    def forward(self, x, rms_w, mm_w):
        x = self.rmsnorm(x, rms_w)
        return torch.matmul(x, mm_w)


model = RMSNorm_Matmul()

i = 16
hidden_size = 1024
j = 4096
x = torch.randn(i, hidden_size, dtype=torch.float32)
rms_w = torch.randn(hidden_size, dtype=torch.float32)
mm_w = torch.randn(hidden_size, j, dtype=torch.float32)
example_args = (x, rms_w, mm_w)

dim_i = Dim("i")
dim_hidden_size = Dim("hidden_size")
dim_j = Dim("j")
dynamic_shapes = {
    "x": {0: dim_i, 1: dim_hidden_size},
    # 为 'rms_w' 指定动态维度
    "rms_w": {0: dim_hidden_size},
    # 为 'mm_w' 指定动态维度
    "mm_w": {0: dim_hidden_size, 1: dim_j},
}


exported: torch.export.ExportedProgram = export(
    model, args=example_args, dynamic_shapes=dynamic_shapes
)
#
print(exported)
