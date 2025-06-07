from Aipiler.dsl import map, reduce, unary, einsum_env, einsum
from Aipiler.tensor import FakeTensor
from Aipiler.dim import create_dim, create_dims
from Aipiler.datatype import DataType, f32
from Aipiler import aot
import iree.runtime as rt
import logging
import numpy as np
import unittest


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@einsum
def reduce_max(A: FakeTensor):
    B = reduce(A, "ij -> i", ["j"], "max")
    return B


@einsum
def matmul(A: FakeTensor, B: FakeTensor):
    C = map(A, B, "ik, kj -> ikj", ["k"], "*")
    D = reduce(C, "ikj -> ij", ["k"], "+")
    E = reduce_max(D)
    return E


A = FakeTensor(create_dims(3, 4), f32)
B = FakeTensor(create_dims(4, 5), f32)
graph = einsum_env.compile(matmul, [A, B])
print("Graph: \n")
print(graph)
print("\n")
exported = aot.export(graph)
print("MLIR: \n")
exported.print_readable()
print("\n")
compiled_binary = exported.compile(save_to=None)


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
    x = np.random.rand(3, 4).astype(np.float32)
    y = np.random.rand(4, 5).astype(np.float32)
    y = vmm.main(x, y)
    logger.info(f"Inference result: {y.to_host()}")
    return y.to_host()


class ModelTest(unittest.TestCase):
    def test_mlp_export_simple(self) -> None:
        output = run_inference()

        self.assertIsNotNone(output, "inference output should not be None")
        self.assertEqual(
            output.shape,
            (3,),
            "output shape doesn't match the expected (3, 5)",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Run unit tests
    unittest.main()
