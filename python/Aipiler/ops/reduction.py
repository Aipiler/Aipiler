from Aipiler.primitive import EinsumPrimitive, reduce, map, populate, unary
from Aipiler.tensor import Tensor
from Aipiler.basic_operator import operator_registry, BaseOperator
from typing import List, Union, Sequence
from functools import partial


# TODO:
def reduction(x: Tensor, dim: List[int], keepdim: bool, op: BaseOperator) -> Tensor:
    l = len(x.symbolic_shape)
    einsum_alphabet = "abcdefghijklmnopqrstuvwxyz"
    assert l < len(einsum_alphabet)
    letters = einsum_alphabet[:l]
    dim = [d + l if d < 0 else d for d in dim]
    assert all(0 <= d < l for d in dim), "Invalid dimension for reduction"
    assert len(dim) > 0, "At least one dimension must be specified for reduction"
    assert len(dim) < l, "Cannot reduce all dimensions"

    reduce_tensor = reduce(
        x,
        "{rhs_letters} -> {lhs_letters}".format(
            rhs_letters="".join([letters[i] for i in dim]),
            lhs_letters="".join([letters[i] for i in range(l) if i not in dim]),
        ),
        [letters[i] for i in range(l) if i in dim],
        op,
    )

    if keepdim is True:
        pass
    else:
        pass


def reduce_sum(x: Tensor, dim: List[int], keepdim: bool = False) -> Tensor:
    pass


def reduce_mean():
    pass


def reduce_min():
    pass


def reduce_max():
    pass
