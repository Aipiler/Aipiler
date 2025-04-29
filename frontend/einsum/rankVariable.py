from typing import List, Optional, Union, Callable, Dict, Any, Tuple, Set
from .tensor import Tensor, TensorRank
from .range import Range, CompoundRange
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .constraint import Constraint

# --- Rank Variable  ---


class RankVariable:
    """Represents a basic rank variable (e.g., 's', 'd')."""

    def __init__(
        self,
        name: str = "",
    ):
        """Initialize a rank variable with a name."""
        self.name = name
        self.static_ranges = CompoundRange()
        self.constraints: List[Constraint] = []

    def add_static_range(self, range: Range):
        """添加静态范围"""
        self.static_ranges.add_range(range)

    def get_static_range(self) -> CompoundRange:
        """获取基于静态约束的范围"""
        return self.static_ranges

    def add_constraint(self, constraint: Constraint):
        """添加约束"""
        self.constraints.append(constraint)

    def setName(self, name: str):
        """Set the name of the rank variable."""
        self.name = name

    def getName(self) -> str:
        """Get the name of the rank variable."""
        return self.name

    def __repr__(self):
        return self.name
