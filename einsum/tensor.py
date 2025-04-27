import inspect
from typing import List, Optional, Union, Callable, Dict, Any, Tuple
from enum import Enum, auto
import operator
from abc import ABC, abstractmethod


class Dtype(Enum):
    """Represents the data type of a tensor."""

    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    # Add more types as needed


# TODO: Empty Value
class Empty(ABC):
    pass


class ZeroEmpty(Empty):
    def __init__(self):
        pass

    def __repr__(self):
        return "0"


class Tensor:
    """Represents a tensor with a name and its rank expressions."""

    def __init__(
        self,
        name: str,
        shape: Tuple[int],
        dtype: Dtype = Dtype.FLOAT,
        empty: Empty = ZeroEmpty(),
    ):
        self.name = name
        self.rank = len(shape)
        self.shape = shape
        self.dtype = dtype
        self.empty = empty

    def get_rank(self) -> int:
        """Get the rank of the tensor."""
        return self.rank

    def get_name(self) -> str:
        """Get the name of the tensor."""
        return self.name

    def get_dtype(self) -> Dtype:
        """Get the data type of the tensor."""
        return self.dtype

    def get_empty(self) -> Empty:
        """Get the empty value of the tensor."""
        return self.empty

    def get_i_shape(self, i: int) -> int:
        """Get the i-th of the tensor shape."""
        return self.shape[i]

    def __repr__(self):
        pass
