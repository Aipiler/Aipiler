from Aipiler.primitive import EinsumPrimitive, reduce, map
from Aipiler.tensor import Tensor
from Aipiler.basic_operator import BaseOperator, operator_registry


def matmul(a: Tensor, b: Tensor):
    t = map(a, b, "ik, kj -> ikj", "k", operator_registry.get("mul"))
    c = reduce(t, "ikj -> ij", "k", operator_registry.get("sum"))
    return c
