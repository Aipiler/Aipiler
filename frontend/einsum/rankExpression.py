from typing import List, Optional, Union, Callable, Dict, Any, Tuple, Set
from .tensor import Tensor, TensorRank
from .range import Range, CompoundRange
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from .term import AffineTerm

if TYPE_CHECKING:
    from .rankVariable import RankVariable


# --- Rank Expressions ---


class RankExpression(ABC):
    """Base class for all forms of rank specifications in tensor subscripts."""

    def __init__(self, variables: List["RankVariable"]):
        self.variables = variables  # The core variable (e.g., 's' in 's:s<d' or 's+5')

    def get_rank_variables(self) -> List["RankVariable"]:
        """Get the core rank variable."""
        return self.variables

    def __repr__(self):
        pass


class SimpleRankExpression(RankExpression):
    """Represents a simple rank expression (e.g., 's', 'd')."""

    def __init__(self, rankVariable: "RankVariable"):
        """Initialize a simple rank expression."""
        super().__init__([rankVariable])
        self.rankVariable = rankVariable

    def get_rank_variable(self) -> "RankVariable":
        """Get the rank variable."""
        return self.rankVariable

    def __repr__(self):
        """String representation of the simple rank expression."""
        return str(self.rankVariable)


# --- Affine Expressions ---


class AffineRankExpression(RankExpression):
    """Base class for affine expressions."""

    def __init__(self, affineTerm: AffineTerm):
        """Initialize an affine rank expression."""
        super().__init__(affineTerm.get_variables())
        self.affineTerm = affineTerm

    def get_affine_term(self) -> AffineTerm:
        """Get the affine term."""
        return self.affineTerm

    def __repr__(self):
        """String representation of the simple rank expression."""
        return f"{self.affineTerm}"


# TODO: 暂时不支持非affine映射rank
class NonAffineRankRank(RankExpression):
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

    # TODO: 测试AffineRankExpression
    # Example: Convolution
