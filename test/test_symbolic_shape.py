import torch
from torch import nn
from torch.export import export, Dim
import Aipiler
from Aipiler.context import Context


class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return torch.matmul(x, weight)


X = torch.randn(16, 1024, dtype=torch.float32)
W = torch.randn(1024, 4096, dtype=torch.float32)
example_args = (X, W)


model = Matmul()

exported_program: torch.export.ExportedProgram = export(model, args=example_args)
print("Exported Program:", exported_program)
print("Graph: ", exported_program.graph)
print("Graph_signature: ", exported_program.graph_signature)
print("State_dict: ", exported_program.state_dict)
print("Range_constraints: ", exported_program.range_constraints)


g = Context().compile(exported_program)
print("Compiled Graph:", g)
