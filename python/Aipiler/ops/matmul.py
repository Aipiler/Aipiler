from Aipiler.primitive import EinsumPrimitive, reduce, map
from Aipiler.tensor import Tensor
from Aipiler.basic_operator import BasicOperator

def matmul(a: Tensor, b: Tensor):
    t = map(a, b, "ik, kj -> ikj", "k", BasicOperator.MUL)
    c = reduce(t, "ikj -> ij", "k", BasicOperator.ADD)
    return c