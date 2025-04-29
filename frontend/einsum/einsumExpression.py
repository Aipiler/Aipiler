from .rankExpression import (
    RankExpression,
    SimpleRankExpression,
    AffineRankExpression,
    NonAffineRankRank,
)
from .rankVariable import RankVariable
from .term import AffineTerm, VarTerm
from .operators.compute import ComputeOperator
from .operators.unary import UnaryOperator
from .operators.coordinate import CoordinateOperator
from .operators.merge import MergeOperator
from .tensor import Tensor, TensorRank, Dtype
from .action import Action, MapAction, ReduceAction, PopulateAction
from .range import Range, CompoundRange
from .rankExpression import RankMap
from .constraint import (
    Constraint,
    StaticConstraint,
    DynamicConstraint,
    ComparisonOperator,
)
from typing import List, Optional, Tuple, Dict, Any, Set, Callable, Union, TypeVar
from abc import ABC, abstractmethod


T = TypeVar("T", bound="EinsumExpression")


class EinsumExpression(ABC):
    """Base class for all Einsum expressions."""

    def __init__(self, output_tensor: Tensor, input_tensors: List[Tensor]):
        self.output_tensor = output_tensor
        self.input_tensors = input_tensors

    @abstractmethod
    def __repr__(self) -> str:
        """Return a developer-friendly string representation of the expression."""
        pass

    def __str__(self) -> str:
        """Return a user-friendly string representation of the expression."""
        return self.__repr__()

    @abstractmethod
    def clone(self: T) -> T:
        """Create a deep copy of this expression."""
        pass

    def format_tensor_list(self, tensors: List[Tensor]) -> str:
        """Helper method to format a list of tensors."""
        return ", ".join([str(tensor) for tensor in tensors])


class EinsumEquation(EinsumExpression):
    """Represents an EDGE Einsum expression."""

    def __init__(
        self, output_tensor: Tensor, input_tensors: List[Tensor], rankMap: RankMap
    ):
        # Extract the output tensor and input tensors from the mappings
        super().__init__(
            output_tensor=output_tensor,
            input_tensors=input_tensors,
        )
        self.rankMap = rankMap

        # Validate inputs
        if not output_tensor:
            raise ValueError("Output tensor cannot be None")
        if not input_tensors:
            raise ValueError("Input tensors list cannot be empty")

    def __repr__(self) -> str:
        input_tensors_str = self.format_tensor_list(self.input_tensors)
        return f"{self.__class__.__name__}(output={self.output_tensor}, inputs=[{input_tensors_str}], rankMap={self.rankMap})"

    def clone(self) -> "EinsumEquation":
        """Create a deep copy of this equation."""
        # This is a basic implementation. Subclasses should override with more specific logic.
        return self.__class__(
            output_tensor=self.output_tensor,  # Assuming tensors are immutable
            input_tensors=list(self.input_tensors),
            rankMap=(
                self.rankMap.clone() if hasattr(self.rankMap, "clone") else self.rankMap
            ),
        )

    def gen_iteration_domain(self):
        """Generate the iteration domain for the Einsum equation.

        Returns:
            An object representing the iteration domain.
        """
        # TODO: 实现迭代域生成逻辑
        raise NotImplementedError("This method needs to be implemented")

    # def gen_iteration_domain(self) -> IterationDomain:
    #     """Generate the iteration domain for the Einsum equation."""
    #     # 创建空的迭代域
    #     iteration_domain = IterationDomain()

    #     def add_variable_domain_to_iteration_domain(tensor_mapping: RankMapping):
    #         """Add ranges for the tensor mapping to the iteration domain."""
    #         tensor = tensor_mapping.get_tensor()
    #         for i in range(tensor.get_rank()):
    #             # Get the rank expression for the i-th dimension
    #             rank_expr = tensor_mapping.get_rank_expression(i)
    #             # Add the range for the i-th dimension
    #             if isinstance(rank_expr, SimpleRankExpression):
    #                 # Check if the rank expression is a simple rank expression
    #                 var = rank_expr.get_rank_variables()
    #                 if iteration_domain.has_variable(var):
    #                     # TODO: check the existing variable domain with current range.
    #                     continue
    #                 # Create a new variable domain for the rank variable
    #                 variable_domain = VariableDomain(var)
    #                 variable_domain.add_static_range(
    #                     Range(0, tensor.get_i_shape(i)),
    #                 )
    #             elif isinstance(rank_expr, AffineMappedRankExpression):
    #                 # TODO: 处理仿射映射的情况, ax + b
    #                 pass
    #             else:
    #                 # TODO: 处理类似卷积的情况, s + q
    #                 pass
    #             # Add the variable domain to the iteration domain
    #             iteration_domain.add_variable_domain(variable_domain)

    #     # Add the ranges for tensor mappings
    #     all_tensor_mappings = self.input_tensor_mappings + [self.output_tensor_mapping]
    #     for tensor_mapping in all_tensor_mappings:
    #         add_variable_domain_to_iteration_domain(tensor_mapping)

    #     # Add the ranges for the output tensor mapping
    #     for constraint in self.constraint:
    #         iteration_domain.add_constraint(constraint)

    #     return iteration_domain


class MapEquation(EinsumEquation):
    """Represents a Map equation in the Einsum expression."""

    def __init__(
        self,
        output_tensor: Tensor,
        first_tensor: Tensor,
        second_tensor: Tensor,
        rankMap: RankMap,
        target_ranks: List[RankVariable],
        computeOp: ComputeOperator,
    ):
        super().__init__(
            output_tensor=output_tensor,
            input_tensors=[first_tensor, second_tensor],
            rankMap=rankMap,
        )
        self.first_tensor = first_tensor
        self.second_tensor = second_tensor
        self.target_ranks = target_ranks
        self.computeOp = computeOp

    def __repr__(self) -> str:
        target_ranks_str = ", ".join([str(rank) for rank in self.target_ranks])
        return (
            f"MapEquation(output={self.output_tensor}, "
            f"first={self.first_tensor}, second={self.second_tensor}, "
            f"targetRanks=[{target_ranks_str}], computeOp={self.computeOp}, "
            f"rankMap={self.rankMap})"
        )

    def clone(self) -> "MapEquation":
        """Create a deep copy of this equation."""
        return MapEquation(
            output_tensor=self.output_tensor,
            first_tensor=self.first_tensor,
            second_tensor=self.second_tensor,
            rankMap=(
                self.rankMap.clone() if hasattr(self.rankMap, "clone") else self.rankMap
            ),
            target_ranks=list(self.target_ranks),
            computeOp=self.computeOp,
        )


class ReduceEquation(EinsumEquation):
    """Represents a Reduce equation in the Einsum expression."""

    def __init__(
        self,
        output_tensor: Tensor,
        input_tensor: Tensor,
        rankMap: RankMap,
        target_ranks: List[RankVariable],
        computeOp: ComputeOperator,
    ):
        super().__init__(
            output_tensor=output_tensor,
            input_tensors=[input_tensor],
            rankMap=rankMap,
        )
        self.input_tensor = input_tensor
        self.target_ranks = target_ranks
        self.computeOp = computeOp

    def __repr__(self):
        target_ranks_str = ", ".join([str(rank) for rank in self.target_ranks])
        return (
            f"ReduceEquation(output={self.output_tensor}, "
            f"input={self.input_tensor}, "
            f"targetRanks=[{target_ranks_str}], computeOp={self.computeOp}, "
            f"rankMap={self.rankMap})"
        )

    def clone(self) -> "ReduceEquation":
        """Create a deep copy of this equation."""
        return ReduceEquation(
            output_tensor=self.output_tensor,
            input_tensor=self.input_tensor,
            rankMap=(
                self.rankMap.clone() if hasattr(self.rankMap, "clone") else self.rankMap
            ),
            target_ranks=list(self.target_ranks),
            computeOp=self.computeOp,
        )


class PopulateEquation(EinsumEquation):
    """Represents a Populate equation in the Einsum expression."""

    def __init__(
        self,
        output_tensor: Tensor,
        input_tensor: Tensor,
        rankMap: RankMap,
        target_ranks: List[RankVariable],
        computeOp: ComputeOperator,
        coordinateOp: CoordinateOperator,
    ):
        super().__init__(
            output_tensor=output_tensor,
            input_tensors=[input_tensor],
            rankMap=rankMap,
        )
        self.input_tensor = input_tensor
        self.target_ranks = target_ranks
        self.computeOp = computeOp
        self.coordinateOp = coordinateOp

    def __repr__(self):
        target_ranks_str = ", ".join([str(rank) for rank in self.target_ranks])
        return (
            f"PopulateEquation(output={self.output_tensor}, "
            f"input={self.input_tensor}, "
            f"targetRanks=[{target_ranks_str}], computeOp={self.computeOp}, "
            f"coordinateOp={self.coordinateOp}, rankMap={self.rankMap})"
        )

    def clone(self) -> "PopulateEquation":
        """Create a deep copy of this equation."""
        return PopulateEquation(
            output_tensor=self.output_tensor,
            input_tensor=self.input_tensor,
            rankMap=(
                self.rankMap.clone() if hasattr(self.rankMap, "clone") else self.rankMap
            ),
            target_ranks=list(self.target_ranks),
            computeOp=self.computeOp,
            coordinateOp=self.coordinateOp,
        )


class UnaryEquation(EinsumEquation):
    """Represents a Unary equation in the Einsum expression."""

    def __init__(
        self,
        output_tensor: Tensor,
        input_tensor: Tensor,
        unaryOp: UnaryOperator,
    ):
        super().__init__(
            output_tensor=output_tensor,
            input_tensors=[input_tensor],
            rankMap=None,
        )
        self.input_tensor = input_tensor
        self.unaryOp = unaryOp

    def __repr__(self) -> str:
        return (
            f"UnaryEquation(output={self.output_tensor}, "
            f"input={self.input_tensor}, unaryOp={self.unaryOp})"
        )

    def clone(self) -> "UnaryEquation":
        """Create a deep copy of this equation."""
        return UnaryEquation(
            output_tensor=self.output_tensor,
            input_tensor=self.input_tensor,
            unaryOp=self.unaryOp,
        )


class EinsumCascade(EinsumExpression):
    """Represents a cascade of Einsum equations."""

    def __init__(
        self,
        output_tensor: Tensor,
        input_tensors: List[Tensor],
        equations: List[EinsumEquation],
    ):
        super().__init__(
            output_tensor=output_tensor,
            input_tensors=input_tensors,
        )
        self.equations = equations

    def __repr__(self) -> str:
        input_tensors_str = self.format_tensor_list(self.input_tensors)
        equations_str = ", ".join([repr(eq) for eq in self.equations])
        return (
            f"EinsumCascade(output={self.output_tensor}, "
            f"inputs=[{input_tensors_str}], "
            f"equations=[{equations_str}])"
        )

    def clone(self) -> "EinsumCascade":
        """Create a deep copy of this cascade."""
        return EinsumCascade(
            output_tensor=self.output_tensor,
            input_tensors=list(self.input_tensors),
            equations=[eq.clone() for eq in self.equations],
        )


class EinsumIteration(EinsumCascade):
    """Represents an iteration in the Einsum expression."""

    def __init__(
        self,
        output_tensor: Tensor,
        input_tensors: List[Tensor],
        equations: List[EinsumEquation],
        generative_rank: RankVariable,
    ):
        super().__init__(
            output_tensor=output_tensor,
            input_tensors=input_tensors,
            equations=equations,
        )
        self.generative_rank = generative_rank

    def __repr__(self) -> str:
        input_tensors_str = self.format_tensor_list(self.input_tensors)
        equations_str = ", ".join([repr(eq) for eq in self.equations])
        return (
            f"EinsumIteration(output={self.output_tensor}, "
            f"inputs=[{input_tensors_str}], "
            f"equations=[{equations_str}], "
            f"generative_rank={self.generative_rank})"
        )

    def clone(self) -> "EinsumIteration":
        """Create a deep copy of this iteration."""
        return EinsumIteration(
            output_tensor=self.output_tensor,
            input_tensors=list(self.input_tensors),
            equations=[eq.clone() for eq in self.equations],
            generative_rank=self.generative_rank,
        )


if __name__ == "__main__":
    # Example usage

    # maxPooling 2d
    pooling_size = 2
    stride = 2
    tensor_input = Tensor("A", (10, 10))
    tensor_output = Tensor("B", (5, 5))
    rankMap = RankMap()
    varM = RankVariable("m")
    varN = RankVariable("n")
    varK = RankVariable("k")
    varK.add_constraint(
        StaticConstraint(varK, ComparisonOperator.LESS_THAN, pooling_size)
    )
    rankMap.add_mapping(
        tensor_input.get_i_rank(0),
        AffineRankExpression(
            affineTerm=AffineTerm(pooling_size, VarTerm(varM, stride), VarTerm(varK))
        ),
    )
    rankMap.add_mapping(
        tensor_input.get_i_rank(1),
        AffineRankExpression(
            affineTerm=AffineTerm(pooling_size, VarTerm(varN, stride), VarTerm(varK))
        ),
    )
    rankMap.add_mapping(
        tensor_output.get_i_rank(0),
        AffineRankExpression(
            affineTerm=AffineTerm(pooling_size, VarTerm(varM, stride))
        ),
    )
    rankMap.add_mapping(
        tensor_output.get_i_rank(1),
        AffineRankExpression(
            affineTerm=AffineTerm(pooling_size, VarTerm(varM, stride))
        ),
    )
    maxPooling = ReduceEquation(
        tensor_output, tensor_input, rankMap, [varK], ComputeOperator.MAX
    )
    print(maxPooling)
