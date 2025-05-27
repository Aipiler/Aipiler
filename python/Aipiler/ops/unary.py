from Aipiler.datatype import DataType
from Aipiler.primitive import EinsumPrimitive, EinsumBuilder
from Aipiler.tensor import Tensor, Dtype, DtypeMapper
from Aipiler.basic_operator import operator_registry


def to_dtype(tensor: Tensor, dtype: DataType) -> Tensor:
    """
    Converts a tensor to the specified dtype.

    Args:
        tensor (Tensor): The input tensor.
        dtype (str): The target dtype.

    Returns:
        Tensor: The tensor converted to the specified dtype.
    """

    return EinsumBuilder.unary(tensor, operator_registry.get(f"to_{dtype.name}"))


def neg(tensor: Tensor) -> Tensor:
    return EinsumBuilder.unary(tensor, operator_registry.get("neg"))


def abs(tensor: Tensor) -> Tensor:
    return EinsumBuilder.unary(tensor, operator_registry.get("abs"))


def pow(tensor: Tensor, exponent: float) -> Tensor:
    """
    Raises each element of the tensor to the specified exponent.

    Args:
        tensor (Tensor): The input tensor.
        exponent (float): The exponent to raise each element to.

    Returns:
        Tensor: A new tensor with each element raised to the specified exponent.
    """
    return EinsumBuilder.unary(tensor, operator_registry.get(f"pow_{exponent}"))


def relu(tensor: Tensor) -> Tensor:
    """
    Applies the ReLU activation function to the tensor.

    Args:
        tensor (Tensor): The input tensor.

    Returns:
        Tensor: A new tensor with ReLU applied.
    """
    return EinsumBuilder.unary(tensor, operator_registry.get("relu"))
