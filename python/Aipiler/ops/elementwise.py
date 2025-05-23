from Aipiler.primitive import EinsumPrimitive, reduce, map
from Aipiler.tensor import Tensor
from Aipiler.basic_operator import BasicOperator


def add(a: Tensor, b: Tensor)->Tensor:
    return map(a, b, "i, i -> i", "i", BasicOperator.ADD)


def sub(a: Tensor, b: Tensor)->Tensor:
    return map(a, b, "i, i -> i", "i", BasicOperator.SUB)


def mul(a: Tensor, b: Tensor)->Tensor:
    return map(a, b, "i, i -> i", "i", BasicOperator.ADD)


def div(a: Tensor, b: Tensor)->Tensor:
    return map(a, b, "i, i -> i", "i", BasicOperator.ADD)