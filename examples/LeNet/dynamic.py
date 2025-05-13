import torch
from torch.export import Dim, export


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU())
        self.branch2 = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU())
        self.buffer = torch.ones(32)

    def forward(self, x1, x2):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        return (out1 + self.buffer, out2)


example_args = (torch.randn(32, 64), torch.randn(32, 128))

# Create a dynamic batch size
batch = Dim("batch")
# Specify that the first dimension of each input is that batch size
dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

exported_program: torch.export.ExportedProgram = export(
    M(), args=example_args, dynamic_shapes=dynamic_shapes
)
print(exported_program)
