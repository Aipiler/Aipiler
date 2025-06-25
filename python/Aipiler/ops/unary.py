from Aipiler.datatype import DataType
from Aipiler.primitive import EinsumPrimitive, EinsumBuilder
from Aipiler.tensor import Tensor, FakeTensor
from Aipiler.basic_operator import operator_registry
from Aipiler import dsl


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


def neg(tensor: FakeTensor) -> FakeTensor:
    return EinsumBuilder.unary(tensor, operator_registry.get("neg"))


def abs(tensor: FakeTensor) -> FakeTensor:
    return EinsumBuilder.unary(tensor, operator_registry.get("abs"))


# def pow(tensor: FakeTensor, exponent: float) -> FakeTensor:
#     """
#     Raises each element of the tensor to the specified exponent.

#     Args:
#         tensor (Tensor): The input tensor.
#         exponent (float): The exponent to raise each element to.

#     Returns:
#         Tensor: A new tensor with each element raised to the specified exponent.
#     """
#     return EinsumBuilder.unary(tensor, operator_registry.get(f"pow_{exponent}"))


def relu(tensor: FakeTensor) -> FakeTensor:
    """
    Applies the ReLU activation function to the tensor.

    Args:
        tensor (Tensor): The input tensor.

    Returns:
        Tensor: A new tensor with ReLU applied.
    """
    return dsl.unary(tensor, "relu")


def rsqrt(tensor: FakeTensor) -> FakeTensor:
    return dsl.unary(tensor, "rsqrt")
