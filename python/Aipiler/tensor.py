from Aipiler.dim import Dim
from Aipiler.datatype import DataType
from typing import Optional, List
import torch


class Tensor:
    def __init__(self, symbolic_shape: List[Dim], dtype: DataType, trace = None) -> None:
        from Aipiler.primitive import EinsumPrimitive
        
        self.symbolic_shape = symbolic_shape
        self.dtype = dtype
        self._trace : Optional[EinsumPrimitive] = trace

    @property
    def dim(self):
        return len(self.symbolic_shape)
    
    @property
    def shape(self):
        return self.symbolic_shape


def from_torch(tensor: torch.Tensor):
    # TODO: datatype of tensor
    assert isinstance(tensor, torch.Tensor)
    dim = tensor.dim()
    return Tensor([Dim() for _ in range(dim)], None)
