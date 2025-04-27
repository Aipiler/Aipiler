import inspect
from typing import List, Optional, Union, Callable, Dict, Any, Tuple
from enum import Enum, auto
import operator
from abc import ABC, abstractmethod

# --- Rank Variable  ---


class RankVariable:
    """Represents a basic rank variable (e.g., 's', 'd')."""

    def __init__(
        self,
        name: str = "",
    ):
        """Initialize a rank variable with a name."""
        self.name = name

    def setName(self, name: str):
        """Set the name of the rank variable."""
        self.name = name

    def getName(self) -> str:
        """Get the name of the rank variable."""
        return self.name

    def __repr__(self):
        return self.name


# --- Rank Expressions ---


class RankExpression(ABC):
    """Base class for all forms of rank specifications in tensor subscripts."""

    def __init__(self, variable: RankVariable):
        self.variable = variable  # The core variable (e.g., 's' in 's:s<d' or 's+5')

    def get_rank_variables(self) -> RankVariable:
        """Get the core rank variable."""
        return self.variable

    @abstractmethod
    def __repr__(self):
        pass

    def check_condition(self) -> bool:
        """Check if the rank expression meets its conditions."""
        # Placeholder for condition checking logic
        return True


class SimpleRankExpression(RankExpression):
    """Represents a simple rank (e.g., 's', 'd')."""

    def __init__(self, variable: RankVariable):
        super().__init__(variable)

    def __repr__(self):
        return f"{self.variable}"


class AffineMap:
    """Represents an affine mapping (e.g., 's+5')."""

    def __init__(self, factor: int, offset: int):
        self.factor = factor
        self.offset = offset  # e.g., 5


class AffineMappedRankExpression(RankExpression):
    """A rank expression that uses an affine mapping (e.g., 's+5')."""

    def __init__(self, variable: RankVariable, affine_map: AffineMap):
        super().__init__(variable)
        self.affine_map = affine_map  # e.g., AffineMap(1, 5)

    def __repr__(self):
        pass


# TODO: 暂时不支持非affine映射
class NonAffineMap:
    pass


# TODO: 暂时不支持非affine映射rank
class NonAffineMappedRank(RankExpression):
    pass
