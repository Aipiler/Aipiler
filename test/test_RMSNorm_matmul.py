import iree.runtime as rt
import logging
import numpy as np
import torch
import torch.nn as nn
import unittest
import os
from Aipiler import aot
from torch.export import export, Dim


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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

k = Dim("k")
dynamic_shapes = {"x": {1: k}, "weight": {0: k}}
model = RMSNorm_Matmul(hidden_size=1024)

# pytorch
exported_program = export(model, example_args, dynamic_shapes)
# print("Exported Program:", exported_program)
# print("Graph: ", exported_program.graph)
# print("Graph_signature: ", exported_program.graph_signature)
# print("State_dict: ", exported_program.state_dict)
# print("Range_constraints: ", exported_program.range_constraints)
# with open("exported_graph.txt", "w", encoding="utf-8") as f:
#     f.write(str(exported_program))

# Aipiler
# exported = aot.export(model, args=example_args)
# exported.print_readable()
# compiled_binary = exported.compile(save_to=None)


def run_inference() -> np.ndarray:
    """
    Runs inference on the compiled model.

    Returns:
        np.ndarray: The result of inference as a NumPy array.
    """
    config = rt.Config("local-task")
    vmm = rt.load_vm_module(
        rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
        config,
    )
    x = np.random.rand(97, 8).astype(np.float32)
    y = vmm.main(x)
    logger.debug(f"Inference result: {y.to_host()}")
    return y.to_host()


class ModelTest(unittest.TestCase):
    def test_mlp_export_simple(self) -> None:
        output = run_inference()

        self.assertIsNotNone(output, "inference output should not be None")
        self.assertEqual(
            output.shape,
            (97, 2),
            "output shape doesn't match the expected (97, 2)",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Run unit tests
    # unittest.main()
