from Aipiler.dsl import map, reduce, unary, einsum_env, einsum
from Aipiler.tensor import FakeTensor
from Aipiler.dim import create_dim, create_dims
from Aipiler.datatype import DataType, f32


@einsum
def test(A: FakeTensor, B: FakeTensor):
    C = map(A, B, "ik, kj -> ikj", ["k"], "*")
    D = reduce(C, "ikj -> ij", ["k"], "+")
    return D


A = FakeTensor(create_dims(3, 4), f32)
B = FakeTensor(create_dims(4, 5), f32)
graph = einsum_env.compile(test, [A, B])
print(graph)
