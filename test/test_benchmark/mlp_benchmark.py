# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.runtime as rt
import logging
import numpy as np
import torch
import torch.nn as nn
import unittest

from Aipiler import aot

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model class.
    Defines a neural network with four linear layers and sigmoid activations.
    """

    def __init__(self) -> object:
        super().__init__()
        # Define model layers
        self.layer0 = nn.Linear(8, 8, bias=True)
        self.layer1 = nn.Linear(8, 4, bias=True)
        self.layer2 = nn.Linear(4, 2, bias=True)
        self.layer3 = nn.Linear(2, 2, bias=True)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after forward pass.
        """
        out = torch.matmul(x, weight)
        return out


if __name__ == "__main__":
    model = MLP()
    X = torch.randn(2, 2, dtype=torch.float32)
    W = torch.randn(2, 2, dtype=torch.float32)
    example_args = (X, W)
    exported = aot.export(model, args=example_args)
    exported.print_readable()

    exported.compile(save_to="./mlp_O1.vmfb", target_backend="rvv")
