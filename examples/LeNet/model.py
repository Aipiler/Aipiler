import torch
import torch.nn as nn


import torch
from torch.export import export


# Simple module for demonstration
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv(x)
        return self.maxpool(self.relu(a))


example_args = (torch.randn(1, 3, 256, 256),)

exported_program: torch.export.ExportedProgram = export(M(), args=example_args)
# print(exported_program.module())
print(exported_program.graph)

for node in exported_program.graph.nodes:
    print(node)
    if node.op == "placeholder":
        print("placeholder")
        print(node.args)
    elif node.op == "call_function":
        print("call_function")
        print(node.args)
        print(node.meta["val"])
    elif node.op == "get_attr":
        print("get_attr")
        print(node.args)
    else:
        print("output")
        print(node.args)

# print(exported_program)
