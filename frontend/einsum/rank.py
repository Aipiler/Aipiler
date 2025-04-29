from typing import List, Optional, Union, Callable, Dict, Any, Tuple, Set
from .tensor import Tensor, TensorRank
from .range import Range, CompoundRange
from typing import TYPE_CHECKING
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
        self.static_ranges = CompoundRange()
        self.constraints: List["Constraint"] = []

    def add_static_range(self, range: Range):
        """添加静态范围"""
        self.static_ranges.add_range(range)

    def get_static_range(self) -> CompoundRange:
        """获取基于静态约束的范围"""
        return self.static_ranges

    def add_constraint(self, constraint: "Constraint"):
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


# --- Rank Mapping ---


class RankMap:

    def __init__(self):
        """Initialize a rank map."""
        self.rank_map: Dict[TensorRank, RankExpression] = {}

    def add_mapping(self, tensor_rank: TensorRank, rank_expression: RankExpression):
        """Add a mapping from a tensor rank to a rank expression."""
        if tensor_rank in self.rank_map:
            raise ValueError(f"Mapping for {tensor_rank} already exists.")
        self.rank_map[tensor_rank] = rank_expression

    def get_mapping(self, tensor_rank: TensorRank) -> Optional[RankExpression]:
        """Get the rank expression for a given tensor rank."""
        return self.rank_map.get(tensor_rank, None)

    def __repr__(self):
        """String representation of the rank map."""
        return "\n".join(
            [
                f"{tensor_rank}: {rank_expression}"
                for tensor_rank, rank_expression in self.rank_map.items()
            ]
        )


if __name__ == "__main__":

    # Example: Matmul
    tensorA = Tensor("A", (3, 4))
    tensorB = Tensor("B", (4, 5))
    tensorC = Tensor("C", (3, 5))

    # init RankVariable
    varM = RankVariable("m")
    varK = RankVariable("k")
    varN = RankVariable("n")

    rankMap = RankMap()
    rankMap.add_mapping(tensorA.get_i_rank(0), SimpleRankExpression(varM))
    rankMap.add_mapping(tensorA.get_i_rank(1), SimpleRankExpression(varK))
    rankMap.add_mapping(tensorB.get_i_rank(0), SimpleRankExpression(varK))
    rankMap.add_mapping(tensorB.get_i_rank(1), SimpleRankExpression(varN))
    rankMap.add_mapping(tensorC.get_i_rank(0), SimpleRankExpression(varM))
    rankMap.add_mapping(tensorC.get_i_rank(1), SimpleRankExpression(varN))

    print(rankMap)

    # TODO: 测试AffineMappedRankExpression
    # Example: Convolution
