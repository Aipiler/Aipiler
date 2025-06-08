from Aipiler.dsl import map, reduce, unary, einsum_env, einsum
from Aipiler.tensor import FakeTensor, FakeScalar
from Aipiler.dim import create_dim, create_dims
from Aipiler.datatype import DataType, f32
from Aipiler import aot


@einsum
def tensor_scalar_map(A: FakeTensor, B: FakeScalar):
    return map(A, B, "ab, _ -> ab", ["a", "b"], "+")


def test_scalar_from_constant():
    A = FakeTensor(create_dims(12, 15), f32)
    B = FakeScalar(10, f32)
    graph = einsum_env.compile(tensor_scalar_map, [A, B])
    print(graph)
    print("\n")
    exported = aot.export(graph)
    print("MLIR: \n")
    exported.print_readable()
    print("\n")
    # compiled_binary = exported.compile(save_to=None)


def test_scalar_from_dim():
    A = FakeTensor(create_dims(12, 15), f32)
    B = FakeScalar(A.symbolic_shape[0], f32)
    graph = einsum_env.compile(tensor_scalar_map, [A, B])
    print(graph)
    print("\n")
    exported = aot.export(graph)
    print("MLIR: \n")
    exported.print_readable()
    print("\n")
    compiled_binary = exported.compile(save_to=None)


def test_scalar_from_dyn_dim():
    A = FakeTensor(create_dims("A.dim0", 15), f32)
    B = FakeScalar(A.symbolic_shape[0], f32)
    graph = einsum_env.compile(tensor_scalar_map, [A, B])
    print(graph)
    print("\n")
    exported = aot.export(graph)
    print("MLIR: \n")
    exported.print_readable()
    print("\n")
    compiled_binary = exported.compile(save_to=None)


test_scalar_from_constant()
