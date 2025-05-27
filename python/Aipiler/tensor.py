from Aipiler.dim import Dim
from Aipiler.datatype import DataType
from typing import Optional, List, Dict
import torch
from enum import Enum
import Aipiler


class Tensor:
    def __init__(self, symbolic_shape: List[Dim], dtype: DataType, trace=None) -> None:
        from Aipiler.primitive import EinsumPrimitive

        self.symbolic_shape = symbolic_shape
        self.dtype = dtype
        self._trace: Optional[EinsumPrimitive] = trace

    @property
    def dim(self):
        return len(self.symbolic_shape)

    @property
    def shape(self):
        return self.symbolic_shape


def from_torch_tensor(tensor: torch.Tensor):
    """From `torch.Tensor` to `Aipiler.Tensor`"""
    from Aipiler.datatype.type_mapper import DtypeMapper

    assert isinstance(tensor, torch.Tensor)
    dim = tensor.dim()
    dtype = DtypeMapper.from_pytorch(tensor.dtype)
    return Tensor([Dim() for _ in range(dim)], dtype, None)
