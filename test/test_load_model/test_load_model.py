from Aipiler import *
import os
import json
from safetensors.torch import load_file
import iree.runtime as rt
import numpy as np


class Linear(Module):
    def __init__(self, in_feature: int, out_feature: int):
        super().__init__()
        self.weight = Parameter(dims(out_feature, in_feature))
        self.bias = Parameter(dims(out_feature))

    def forward(self, x):
        t0 = map(x, self.weight, "ik, jk -> ikj", "*")
        t1 = reduce(t0, "ikj -> ij", "k", "+")
        y = map(t1, self.bias, "ij, j -> ij", "+")
        return y


class _3MM(Module):
    def __init__(self):
        super().__init__()
        input_output_dim = 2
        num_layers = 3
        
        self.layers = ModuleList([Linear(input_output_dim, input_output_dim) for _ in range(num_layers)])
        
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_():
    model = _3MM()
    model = load_from_safetensor(model, "./model.safetensors")
    for _ in model.named_parameters():
        print(_)