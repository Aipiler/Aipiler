from .operators import ComputeOperator, MergeOperator, CoordinateOperator
from .rank import (
    RankVariable,
    RankExpression,
    SimpleRankExpression,
    AffineMappedRankExpression,
)
from .tensor import Tensor, TensorRank, Dtype
from .action import Action, MapAction, ReduceAction, PopulateAction
from .range import Range, CompoundRange
from .rank import RankMap
from .constraint import Constraint, StaticConstraint, DynamicConstraint
from typing import List, Optional, Tuple, Dict, Any, Set, Callable, Union
from abc import ABC, abstractmethod
from enum import Enum, auto
import operator


class EinsumExpression(ABC):
    """Base class for all Einsum expressions."""

    def __init__(self, output_tensor: Tensor, input_tensors: List[Tensor]):
        self.output_tensor = output_tensor
        self.input_tensors = input_tensors

    @abstractmethod
    def __repr__(self):
        pass


class EinsumEquation(ABC, EinsumExpression):
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

    # TODO: 通过rankMap生成迭代空间
    # 迭代空间是一个RankVariable的集合。
    # 数据空间是一个TensorRank的集合。
    def gen_iteration_domain(self):
        pass

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
    ):
        super().__init__(
            output_tensor=output_tensor,
            input_tensors=[first_tensor, second_tensor],
            rankMap=rankMap,
        )
        self.first_tensor = first_tensor
        self.second_tensor = second_tensor
        self.target_ranks = target_ranks

    def __repr__(self):
        pass


class ReduceEquation(EinsumEquation):
    """Represents a Reduce equation in the Einsum expression."""

    def __init__(
        self,
        output_tensor: Tensor,
        input_tensor: Tensor,
        rankMap: RankMap,
        target_ranks: List[RankVariable],
    ):
        super().__init__(
            output_tensor=output_tensor,
            input_tensor=[input_tensor],
            rankMap=rankMap,
        )
        self.input_tensor = input_tensor
        self.target_ranks = target_ranks

    def __repr__(self):
        pass


class PopulateEquation(EinsumEquation):
    """Represents a Populate equation in the Einsum expression."""

    def __init__(
        self,
        output_tensor: Tensor,
        input_tensor: Tensor,
        rankMap: RankMap,
        target_ranks: List[RankVariable],
    ):
        super().__init__(
            output_tensor=output_tensor,
            input_tensor=[input_tensor],
            rankMap=rankMap,
        )
        self.input_tensor = input_tensor
        self.target_ranks = target_ranks

    def __repr__(self):
        pass


class EinsumIteration(EinsumExpression):
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
        )
        self.equations = equations
        self.generative_rank = generative_rank

    def __repr__(self):
        pass


if __name__ == "__main__":
    # Example usage
    pass
