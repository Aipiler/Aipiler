import torch
from torch.export import export


class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b


example_args = (torch.randn(10, 10), torch.randn(10, 10))

model = Mod()

exported_program: torch.export.ExportedProgram = export(Mod(), args=example_args)
print("Graph: ", exported_program.graph)
print("Graph_signature: ", exported_program.graph_signature)
print("State_dict: ", exported_program.state_dict)
print("Range_constraints: ", exported_program.range_constraints)
