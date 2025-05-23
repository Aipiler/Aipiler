from enum import Enum
import operator

def pass_through(x):
    return x

def pos(x):
    return +x

def neg(x):
    return -x

class BasicOperator(Enum):
    ADD = (operator.add, "add")
    MUL = (operator.mul, "mul")
    SUB = (operator.sub, "sub")
    DIV = (operator.floordiv, "div")
    MIN = (operator.lt, "min")
    MAX = (operator.ge, "max")
    PASS_THROUGH = (pass_through, "pass_through")
    POS = (pos, "pos")
    NEG = (neg, "neg")
