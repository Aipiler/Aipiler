from .operators import ComputeOperator, MergeOperator, CoordinateOperator
from .rank import (
    RankVariable,
    RankExpression,
    SimpleRankExpression,
    AffineMappedRankExpression,
)
from .tensor import Tensor, TensorRank, Dtype
from .action import Action, MapAction, ReduceAction, PopulateAction
from typing import List, Optional, Tuple, Dict, Any, Set, Callable, Union
from abc import ABC, abstractmethod
from enum import Enum, auto
import operator


class Range:
    """Represents a range of rank variables."""

    def __init__(
        self,
        lower_bound: int = 0,
        upper_bound: int = 0,
    ):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def setUpperBound(self, upper_bound: int):
        """Set the upper bound of the rank variable."""
        self.upper_bound = upper_bound

    def setLowerBound(self, lower_bound: int):
        """Set the lower bound of the rank variable."""
        self.lower_bound = lower_bound

    def getUpperBound(self) -> int:
        """Get the upper bound of the rank variable."""
        return self.upper_bound

    def getLowerBound(self) -> int:
        """Get the lower bound of the rank variable."""
        return self.lower_bound

    def __repr__(self):
        return f"[{self.lower_bound}, {self.upper_bound})"


class ComparisonOperator(Enum):
    """表示Python中所有比较运算符的枚举类"""

    def __init__(self, func: Callable, symbol: str):
        self.func = func  # 操作符对应的函数
        self.symbol = symbol  # 操作符的符号表示

    def __str__(self):
        return self.symbol


EQUAL = ComparisonOperator(operator.eq, "==")
NOT_EQUAL = ComparisonOperator(operator.ne, "!=")
LESS_THAN = ComparisonOperator(operator.lt, "<")
LESS_THAN_OR_EQUAL = ComparisonOperator(operator.le, "<=")
GREATER_THAN = ComparisonOperator(operator.gt, ">")
GREATER_THAN_OR_EQUAL = ComparisonOperator(operator.ge, ">=")


class Constraint:

    def __init__(
        self,
        variable: RankVariable,
        operator: ComparisonOperator,
        right: int | RankVariable,
    ):
        """
        Args:
            variable: 迭代空间变量
            expression: 约束表达式
        """
        self.variable = variable
        self.operator = operator
        self.right = right  # 右侧的值，可以是整数或RankVariable
        self.static = True  # 是否是静态约束
        if isinstance(right, RankVariable):
            self.static = False

    def is_static(self) -> bool:
        """判断约束是否是静态约束"""
        return self.static

    def gen_range(self) -> Range:
        """生成约束对应的范围"""
        if self.static:
            # 静态约束
            if self.operator == EQUAL:
                # TODO: 处理等式的范围
                pass
                # return Range(self.right, self.right + 1)
            elif self.operator == NOT_EQUAL:
                # TODO: 处理不等式的范围
                pass
                # return Range(0, self.right) | Range(self.right + 1, float("inf"))
            elif self.operator == LESS_THAN:
                return Range(0, self.right)
            elif self.operator == LESS_THAN_OR_EQUAL:
                return Range(0, self.right + 1)
            elif self.operator == GREATER_THAN:
                return Range(self.right + 1, float("inf"))
            elif self.operator == GREATER_THAN_OR_EQUAL:
                return Range(self.right, float("inf"))
        else:
            # 动态约束
            pass


class IterationDomain:
    """表示迭代空间的域"""

    def __init__(self, rank_variables: Set[RankVariable]):
        """
        Args:
            rank_variables: 迭代空间变量列表
        """
        self.rank_variables = rank_variables
        self.ranges = {}  # 迭代空间变量的范围
        self.constraints = []  # 约束列表

    def get_rank_variables(self) -> Set[RankVariable]:
        """返回迭代空间变量列表"""
        return self.rank_variables

    def add_range(self, variable: RankVariable, range: Range):
        """添加迭代空间变量的范围"""
        # 检测变量是否在迭代空间变量列表中
        if variable not in self.rank_variables:
            raise ValueError(f"Variable {variable} not in rank variables.")

        # 检测范围是否已经存在
        if variable not in self.ranges:
            self.ranges[variable] = range
        else:
            # 合并范围
            existing_range: Range = self.ranges[variable]
            existing_range.setLowerBound(
                min(existing_range.getLowerBound(), range.getLowerBound())
            )
            existing_range.setUpperBound(
                max(existing_range.getUpperBound(), range.getUpperBound())
            )

    def add_constraint(self, constraint: Constraint):
        """添加约束到迭代空间"""
        # TODO: 检测约束是否与已有约束冲突，是否能合并。
        if constraint.variable not in self.rank_variables:
            raise ValueError(f"Variable {constraint.variable} not in rank variables.")

        if constraint.is_static():
            # 静态约束
            range = constraint.gen_range()
            self.add_range(constraint.variable, range)
        else:
            # 动态约束: s < d
            self.constraints.append(constraint)

    def __repr__(self):
        pass


class TensorMapping:
    """定义张量和迭代空间变量之间的映射关系"""

    def __init__(self, tensor: Tensor, dimension_mappings: Dict[int, RankExpression]):
        """
        Args:
            tensor: 目标张量
            dimension_mappings: 将张量维度索引映射到迭代空间表达式的字典
                               例如 {0: m_expr, 1: n_expr} 表示张量的第一维映射到m，第二维映射到n
        """
        self.tensor = tensor
        self.dimension_mappings = dimension_mappings
        # 检测映射的表达式是否符合张量的维度
        if len(dimension_mappings) != tensor.get_rank():
            raise ValueError(
                f"Dimension mappings length {len(dimension_mappings)} does not match tensor rank {tensor.get_rank()}."
            )

    def get_accessed_variables(self) -> Set[RankVariable]:
        """返回此映射访问的所有迭代变量"""
        variables = set()
        for expr in self.dimension_mappings.values():
            variables.update(expr.get_rank_variables())
        return variables

    def get_tensor(self) -> Tensor:
        """返回映射的张量"""
        return self.tensor

    def get_rank_expression(self, dim: int) -> RankExpression:
        """返回指定维度的RankExpression"""
        if dim not in self.dimension_mappings:
            raise ValueError(f"Dimension {dim} not in dimension mappings.")
        return self.dimension_mappings[dim]

    def __repr__(self):
        mappings = [f"{dim}->{expr}" for dim, expr in self.dimension_mappings.items()]
        return f"TensorMapping({self.tensor.name}: {', '.join(mappings)})"


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
        self,
        output_tensor_mapping: TensorMapping,
        input_tensor_mappings: List[TensorMapping],
        constraint: List[Constraint] = [],
    ):
        self.output_tensor_mapping = output_tensor_mapping
        self.input_tensor_mappings = input_tensor_mappings
        self.constraint = constraint

        # Extract the output tensor and input tensors from the mappings
        super().__init__(
            output_tensor=self.output_tensor_mapping.get_tensor(),
            input_tensors=[
                mapping.get_tensor() for mapping in self.input_tensor_mappings
            ],
        )

    def gen_iteration_domain(self) -> IterationDomain:
        """Generate the iteration domain for the Einsum equation."""
        all_tensor_mappings = self.input_tensor_mappings + [self.output_tensor_mapping]
        rank_variables = Set(
            tensor_mapping.get_accessed_variables()
            for tensor_mapping in all_tensor_mappings
        )

        # Create the iteration domain with the rank variables
        iteration_domain = IterationDomain(rank_variables)

        def add_range_to_domain(tensor_mapping: TensorMapping):
            """Add ranges for the tensor mapping to the iteration domain."""
            tensor = tensor_mapping.get_tensor()
            for i in range(tensor.get_rank()):
                # Get the rank expression for the i-th dimension
                rank_expr = tensor_mapping.get_rank_expression(i)
                # Add the range for the i-th dimension
                if isinstance(rank_expr, SimpleRankExpression):
                    # Check if the rank expression is a simple rank expression
                    iteration_domain.add_range(
                        rank_expr.get_rank_variables(),
                        Range(0, tensor.get_i_shape(i)),
                    )
                elif isinstance(rank_expr, AffineMappedRankExpression):
                    # TODO: 处理仿射映射的情况, ax + b
                    pass
                else:
                    # TODO: 处理类似卷积的情况, s + q
                    pass

        # Add the ranges for tensor mappings
        for tensor_mapping in all_tensor_mappings:
            add_range_to_domain(tensor_mapping)

        # Add the ranges for the output tensor mapping
        for constraint in self.constraint:
            iteration_domain.add_constraint(constraint)

        return iteration_domain


class MapEquation(EinsumEquation):
    """Represents a Map equation in the Einsum expression."""

    def __init__(
        self,
        output_tensor_mapping: TensorMapping,
        first_tensor_mapping: TensorMapping,
        second_tensor_mapping: TensorMapping,
        target_ranks: List[RankVariable],
        constraint: List[Constraint] = [],
    ):
        super().__init__(
            output_tensor_mapping=output_tensor_mapping,
            input_tensor_mappings=[first_tensor_mapping, second_tensor_mapping],
            constraint=constraint,
        )
        self.first_tensor_mapping = first_tensor_mapping
        self.second_tensor_mapping = second_tensor_mapping
        self.target_ranks = target_ranks

    def __repr__(self):
        pass


# TODO
class ReduceEquation(EinsumEquation):
    """Represents a Reduce equation in the Einsum expression."""

    def __init__(
        self,
        output_tensor_mapping: TensorMapping,
        input_tensor_mappings: TensorMapping,
    ):
        super().__init__(
            output_tensor_mapping=output_tensor_mapping,
            input_tensor_mappings=[input_tensor_mappings],
            action=action,
        )


# TODO
class PopulateEquation(EinsumEquation):
    """Represents a Populate equation in the Einsum expression."""

    def __init__(
        self,
        output_tensor_mapping: TensorMapping,
        input_tensor_mappings: TensorMapping,
    ):
        super().__init__(
            output_tensor_mapping=output_tensor_mapping,
            input_tensor_mappings=[input_tensor_mappings],
            action=action,
        )


# TODO
class EinsumIteration(EinsumExpression):
    pass


if __name__ == "__main__":
    # Example usage
    TensorA = Tensor("A", [TensorRank(2), TensorRank(3)])
    TesnorB = Tensor("B", [TensorRank(3), TensorRank(4)])
    TensorC = Tensor("C", [TensorRank(2), TensorRank(4)])

    # generate RankVariable
    m = RankVariable("m")
    n = RankVariable("n")
    k = RankVariable("k")

    # RankExpression
    m_expr = SimpleRankExpression(m)
    n_expr = SimpleRankExpression(n)
    k_expr = SimpleRankExpression(k)

    # mapping

    matmul = EinsumEquation(
        output_tensor=TensorC,
        input_tensors=[TensorA, TesnorB],
        action=MapAction(target_ranks=[k]),
    )
    print(matmul)
