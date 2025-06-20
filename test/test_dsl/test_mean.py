from Aipiler.dsl import map, reduce, unary, einsum_env, einsum
from Aipiler.tensor import FakeTensor, FakeScalar
from Aipiler.dim import dim, dims
from Aipiler.datatype import DataType, f32
from Aipiler import aot
import iree.runtime as rt
import logging
import numpy as np
import unittest


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@einsum
def mean(A: FakeTensor):
    t0 = reduce(A, "ij->i", "j", "+")
    dim = FakeScalar(
        A.symbolic_shapes[1],
        A.dtype,
    )
    t1 = map(t0, dim, "i, _ -> i", target_dim=["i"], compute_op_str="/")
    return t1


A = FakeTensor(dims(4, 3), f32)
graph = einsum_env.compile(mean, [A])
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
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]).astype(np.float32)
    # x = np.random.rand(10, 3).astype(np.float32)
    y = vmm.main(x)
    logger.info(f"Inference result: {y.to_host()}")
    return y.to_host()


class ModelTest(unittest.TestCase):
    def test_mlp_export_simple(self) -> None:
        output = run_inference()
        print(output)
        # self.assertIsNotNone(output, "inference output should not be None")
        # self.assertEqual(
        #     output.shape,
        #     (3,),
        #     "output shape doesn't match the expected (3, 5)",
        # )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Run unit tests
    unittest.main()
