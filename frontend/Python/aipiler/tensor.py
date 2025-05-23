from aipiler.dim import Dim
from typing import Optional, List



class Tensor:
    def __init__(self, symbolic_shape: List[Dim], trace = None) -> None:
        self.symbolic_shape = symbolic_shape
        self._trace = trace