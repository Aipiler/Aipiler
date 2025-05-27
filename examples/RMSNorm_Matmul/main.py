import torch
from torch import nn
from torch.export import export
import Aipiler


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class RMSNorm_Matmul(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.rmsnorm = Qwen2RMSNorm(hidden_size, eps)

    def forward(self, x, weight):
        x = self.rmsnorm(x)
        return torch.matmul(x, weight)


X = torch.randn(16, 1024, dtype=torch.float32)
W = torch.randn(1024, 4096, dtype=torch.float32)
example_args = (X, W)

model = RMSNorm_Matmul(hidden_size=1024)

exported_program: torch.export.ExportedProgram = export(model, args=example_args)
print("Exported Program:", exported_program)
print("Graph: ", exported_program.graph)
print("Graph_signature: ", exported_program.graph_signature)
print("State_dict: ", exported_program.state_dict)
print("Range_constraints: ", exported_program.range_constraints)


g = Aipiler.compile(exported_program)
print("Compiled Graph:", g)
