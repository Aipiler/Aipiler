from Aipiler.primitive import EinsumPrimitive, EinsumBuilder
from Aipiler.tensor import FakeTensor
from Aipiler.basic_operator import operator_registry, BaseOperator
from typing import List, Union, Sequence
from functools import partial


# TODO:
def reduction(x: FakeTensor, target_dims: List[int], op: BaseOperator) -> FakeTensor:
    l = len(x.symbolic_shapes)
    x_dims = x.symbolic_shapes
    einsum_alphabet = "abcdefghijklmnopqrstuvwxyz"
    assert l < len(einsum_alphabet)
    letters = einsum_alphabet[:l]
    target_dims = [d + l if d < 0 else d for d in target_dims]
    assert all(0 <= d < l for d in target_dims), "Invalid dimension for reduction"
    assert (
        len(target_dims) > 0
    ), "At least one dimension must be specified for reduction"
    assert len(target_dims) < l, "Cannot reduce all dimensions"

    reduce_tensor = EinsumBuilder.reduce(
        x,
        "{lhs_letters} -> {rhs_letters}".format(
            lhs_letters="".join([letters[i] for i in x_dims]),
            rhs_letters="".join([letters[i] for i in range(l) if i not in x_dims]),
        ),
        [letters[i] for i in range(l) if i in target_dims],
        op,
    )

    return reduce_tensor


def reduce_sum(x: FakeTensor, dim: List[int], keepdim: bool = False) -> FakeTensor:
    pass


def reduce_mean():
    pass


def reduce_min():
    pass


def reduce_max():
    pass
