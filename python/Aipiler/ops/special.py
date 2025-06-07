from Aipiler.primitive import EinsumPrimitive, EinsumBuilder
from Aipiler.tensor import Tensor
from Aipiler.basic_operator import ComputeOperator, operator_registry


def matmul(a: Tensor, b: Tensor):
    t = EinsumBuilder.map(a, b, "ik, kj -> ikj", "k", operator_registry.get("mul"))
    c = EinsumBuilder.reduce(t, "ikj -> ij", "k", operator_registry.get("add"))
    return c
