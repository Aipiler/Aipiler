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


# --- Term classes ---


class VarTerm:
    """Represents a variable term with a coefficient (e.g., 2s)."""

    def __init__(self, variable: RankVariable, coefficient: int = 1):
        self.variable = variable
        self.coefficient = coefficient

    def get_variable(self) -> RankVariable:
        """Get the variable associated with this term."""
        return self.variable

    def get_coefficient(self) -> int:
        """Get the coefficient of this term."""
        return self.coefficient

    def __repr__(self):
        if self.coefficient == 1:
            return f"{self.variable}"
        elif self.coefficient == -1:
            return f"-{self.variable}"
        else:
            return f"{self.coefficient}{self.variable}"


class AffineTerm:
    """Represents a term in an affine expression."""

    def __init__(self, constTerm: int = 0, *varTerms: VarTerm):
        """
        Initialize an affine term with a constant and variable terms.

        Args:
            constTerm: The constant term (default: 0)
            *varTerms: Variable number of VarTerm objects
        """
        self.varTerms = list(varTerms)  # 将元组转换为列表以便后续修改
        self.constTerm = constTerm

    def get_var_terms(self) -> List[VarTerm]:
        """Get all variable terms."""
        return self.varTerms

    def get_const_term(self) -> int:
        """Get the constant term."""
        return self.constTerm

    def get_variables(self) -> List[RankVariable]:
        """Get all variables in the affine expression."""
        return [varTerm.get_variable() for varTerm in self.varTerms]

    def add_var_term(self, varTerm: VarTerm):
        """Add a variable term to the affine expression."""
        self.varTerms.append(varTerm)

    def __repr__(self):
        """String representation of the affine term."""
        result = ""

        # Add variable terms
        if self.varTerms:
            result = str(self.varTerms[0])
            for term in self.varTerms[1:]:
                if term.get_coefficient() >= 0:
                    result += f" + {term}"
                else:
                    # For negative coefficients, the minus sign is already included in the term's string representation
                    result += f" {term}"

        # Add constant term if non-zero
        if self.constTerm != 0:
            if self.constTerm > 0:
                prefix = " + " if result else ""
                result += f"{prefix}{self.constTerm}"
            else:
                result += f" - {abs(self.constTerm)}"

        # Return "0" if the expression is empty
        return result if result else "0"


# --- Rank Expressions ---


class RankExpression(ABC):
    """Base class for all forms of rank specifications in tensor subscripts."""

    def __init__(self, variables: List[RankVariable]):
        self.variables = variables  # The core variable (e.g., 's' in 's:s<d' or 's+5')

    def get_rank_variables(self) -> List[RankVariable]:
        """Get the core rank variable."""
        return self.variables

    @abstractmethod
    def __repr__(self):
        pass

    def check_condition(self) -> bool:
        """Check if the rank expression meets its conditions."""
        # Placeholder for condition checking logic
        return True


class SimpleRankExpression(RankExpression):
    """Represents a simple rank expression (e.g., 's', 'd')."""

    def __init__(self, rankVariable: RankVariable):
        """Initialize a simple rank expression."""
        super().__init__([rankVariable])
        self.rankVariable = rankVariable

    def get_rank_variable(self) -> RankVariable:
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
