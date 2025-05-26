from Aipiler.dim import Dim
from typing import Optional, List
import torch


class Tensor:
    def __init__(self, symbolic_shape: List[Dim], trace=None) -> None:
        from Aipiler.primitive import EinsumPrimitive

        self.symbolic_shape = symbolic_shape
        self._trace: Optional[EinsumPrimitive] = trace


def from_torch_tensor(tensor: torch.Tensor) -> Tensor:
    assert isinstance(tensor, torch.Tensor)
    dim = tensor.dim()
    return Tensor([Dim() for _ in range(dim)])
