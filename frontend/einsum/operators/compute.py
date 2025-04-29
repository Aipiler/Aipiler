import operator
from enum import Enum
from typing import Callable


# Predefined compute operators


def pass_through(x):
    return x


class ComputeOperator(Enum):

    ADD = (operator.add, "add")
    MUL = (operator.mul, "mul")
    SUB = (operator.sub, "sub")
    DIV = (operator.floordiv, "div")
    MIN = (operator.lt, "min")
    MAX = (operator.ge, "max")
    PASS_THROUGH = (pass_through, "pass_through")

    def __init__(self, func: Callable, name: str):
        self.func = func  # The actual function for the operator
        self.name = name

    def __repr__(self) -> str:
        return f"ComparisonOperator.{self.name}"

    def __str__(self) -> str:
        return self.symbol
