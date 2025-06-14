from Aipiler.primitive import EinsumPrimitive, EinsumBuilder
from Aipiler.tensor import Tensor
from Aipiler.basic_operator import operator_registry, ComputeOperator
from functools import partial


def binary_elementwise(a: Tensor, b: Tensor, op: ComputeOperator) -> Tensor:
    assert len(a.symbolic_shape) == len(b.symbolic_shape)
    l = len(a.symbolic_shape)
    einsum_alphabet = "abcdefghijklmnopqrstuvwxyz"
    assert l < len(einsum_alphabet)
    letters = einsum_alphabet[:l]

    return EinsumBuilder.map(
        a,
        b,
        "{letters}, {letters} -> {letters}".format(letters=letters),
        list(letters),
        op,
    )


add = partial(binary_elementwise, op=operator_registry.get("add"))
sub = partial(binary_elementwise, op=operator_registry.get("sub"))
mul = partial(binary_elementwise, op=operator_registry.get("mul"))
div = partial(binary_elementwise, op=operator_registry.get("div"))
