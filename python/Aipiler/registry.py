from typing import Dict, Callable, Union
from Aipiler.tensor import Tensor
from Aipiler.datatype import DataType
from Aipiler import ops
import torch


class Registry:
    # registered functions, like torch.add, torch.mul, torch.nn.functional.relu, and torch.ops.aten.cos.
    registered_functions: Dict[Callable, Callable] = {}
    # registered methods, like torch.Tensor.add, torch.Tensor.mul, torch.Tensor.relu.
    registered_methods: Dict[Callable, Callable] = {}


def register_function(func: Union[Callable, str]):
    def decorator(aipiler_func):
        if isinstance(func, str):
            nfunc = eval(func)
        else:
            nfunc = func
        if nfunc not in Registry.registered_functions:
            Registry.registered_functions[nfunc] = aipiler_func
        return aipiler_func

    return decorator


def register_method(method: Callable):
    def decorator(aipiler_method):
        if method not in Registry.registered_methods:
            Registry.registered_methods[method] = aipiler_method
        return aipiler_method

    return decorator


@register_function(torch.ops.aten.matmul.default)
@register_function(torch.matmul)
def matmul(A: Tensor, B: Tensor):
    return ops.special.matmul(A, B)


@register_function(torch.add)
@register_function(torch.ops.aten.add.Tensor)
def add(A, B):
    return ops.add(A, B)


@register_function(torch.ops.aten.to.dtype)
def to_dtype(tensor: Tensor, dtype: DataType) -> Tensor:
    return ops.to_dtype(tensor, dtype)


@register_function(torch.ops.aten.pow.Tensor_Scalar)
def pow(tensor: Tensor, exponent: float) -> Tensor:
    return ops.pow(tensor, exponent)
