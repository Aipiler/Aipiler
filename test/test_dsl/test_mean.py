from Aipiler.dsl import map, reduce, unary, einsum_env, einsum
from Aipiler.tensor import FakeTensor, FakeScalar
from Aipiler.dim import create_dim, create_dims
from Aipiler.datatype import DataType, f32


@einsum
def mean(A: FakeTensor):
    t0 = reduce(A, "ij->i", "j", "+")
    dim = FakeScalar(
        A.symbolic_shape[1],
        A.dtype,
    )
    t1 = map(t0, dim, "i, _ -> i", target_dim=["i"], compute_op_str="/")
    return t1


A = FakeTensor(create_dims(12, 15), f32)
graph = einsum_env.compile(mean, [A])
print("Graph: \n")
print(graph)
print("\n")
