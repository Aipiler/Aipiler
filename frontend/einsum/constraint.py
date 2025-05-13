from .range import Range, CompoundRange
from typing import List, Optional, Tuple, Dict, Any, Set, Callable, Union
from abc import ABC, abstractmethod
from enum import Enum, auto
import operator
from typing import TYPE_CHECKING
from .rankVariable import RankVariable


class ComparisonOperator(Enum):
    """表示Python中所有比较运算符的枚举类"""

    EQUAL = (operator.eq, "==")
    NOT_EQUAL = (operator.ne, "!=")
    LESS_THAN = (operator.lt, "<")
    LESS_THAN_OR_EQUAL = (operator.le, "<=")
    GREATER_THAN = (operator.gt, ">")
    GREATER_THAN_OR_EQUAL = (operator.ge, ">=")

    def __init__(self, func: Callable, symbol: str):
        self.func = func  # 操作符对应的函数
        self.symbol = symbol  # 操作符的符号表示

    def __str__(self):
        return self.symbol


class Constraint(ABC):
    """约束的基类"""

    def __init__(self, variable: RankVariable, operator: ComparisonOperator):
        self.variable = variable
        self.operator = operator

    @abstractmethod
    def is_static(self) -> bool:
        """判断约束是否是静态约束"""
        pass

    @abstractmethod
    def gen_range(self) -> CompoundRange:
        """生成约束对应的范围"""
        pass


class StaticConstraint(Constraint):
    """静态约束：右侧是常量值"""

    def __init__(
        self, variable: RankVariable, operator: ComparisonOperator, value: int
    ):
        super().__init__(variable, operator)
        self.value = value

    def is_static(self) -> bool:
        return True

    def gen_range(self) -> CompoundRange:
        """生成约束对应的范围"""
        result = CompoundRange()

        if self.operator == ComparisonOperator.EQUAL:
            result.add_range(Range(self.value, self.value + 1))
        elif self.operator == ComparisonOperator.NOT_EQUAL:
            # 不等于约束返回两个范围: [0, value) 和 (value, ∞)
            if self.value > 0:
                result.add_range(Range(0, self.value))
            result.add_range(Range(self.value + 1, Range.INFINITY))
        elif self.operator == ComparisonOperator.LESS_THAN:
            result.add_range(Range(0, self.value))
        elif self.operator == ComparisonOperator.LESS_THAN_OR_EQUAL:
            result.add_range(Range(0, self.value + 1))
        elif self.operator == ComparisonOperator.GREATER_THAN:
            result.add_range(Range(self.value + 1, Range.INFINITY))
        elif self.operator == ComparisonOperator.GREATER_THAN_OR_EQUAL:
            result.add_range(Range(self.value, Range.INFINITY))

        return result

    def __repr__(self):
        return f"{self.variable.name} {self.operator} {self.value}"


class DynamicConstraint(Constraint):
    """动态约束：右侧是另一个变量"""

    def __init__(
        self,
        variable: RankVariable,
        operator: ComparisonOperator,
        right_var: RankVariable,
    ):
        super().__init__(variable, operator)
        self.right_var = right_var

    def is_static(self) -> bool:
        return False

    def get_dependent_variable(self) -> RankVariable:
        """获取此约束依赖的变量"""
        return self.right_var

    def __repr__(self):
        return f"{self.variable.name} {self.operator} {self.right_var.name}"


def create_constraint(
    variable: RankVariable,
    operator: ComparisonOperator,
    right: Union[int, RankVariable],
) -> Constraint:
    """创建约束的便捷方法"""

    if isinstance(right, int):
        return StaticConstraint(variable, operator, right)
    elif isinstance(right, RankVariable):
        return DynamicConstraint(variable, operator, right)
    else:
        raise TypeError(f"Unsupported right operand type: {type(right)}")
