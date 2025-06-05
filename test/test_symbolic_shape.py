import torch
from torch import nn
from torch.export import export, Dim
import Aipiler
from Aipiler import aot


class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight):
        return torch.matmul(x, weight)


X = torch.randn(16, 1024, dtype=torch.float32)
W = torch.randn(1024, 4096, dtype=torch.float32)
example_args = (X, W)
k = Dim("k")
dynamic_shapes = {"x": {1: k}, "weight": {0: k}}

model = Matmul()

exported: torch.export.ExportedProgram = aot.export(
    model, args=example_args  # , dynamic_shapes=dynamic_shapes
)
#
exported.print_readable()
# print("Exported Program:", exported_program)
# print("Graph: ", exported_program.graph)
# print("Graph_signature: ", exported_program.graph_signature)
# print("State_dict: ", exported_program.state_dict)
# print("Range_constraints: ", exported_program.range_constraints)


# g = Context(exported_program).compile()
from Aipiler.dynamo_backend import aipiler_backend

# model = torch.compile(model=model, backend=aipiler_backend, fullgraph=True)
model(X, W)
# print("Compiled Graph:", g)
