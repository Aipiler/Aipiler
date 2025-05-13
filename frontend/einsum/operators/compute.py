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
