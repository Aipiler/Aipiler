from Aipiler.primitive import EinsumPrimitive, EinsumBuilder
from Aipiler.tensor import Tensor, FakeTensor, FakeScalar
from Aipiler import dsl
from Aipiler.basic_operator import ComputeOperator, operator_registry


def matmul(lhs: FakeTensor, rhs: FakeTensor):
    t = EinsumBuilder.map(lhs, rhs, "ik, kj -> ikj", "k", operator_registry.get("mul"))
    c = EinsumBuilder.reduce(t, "ikj -> ij", "k", operator_registry.get("add"))
    return c


