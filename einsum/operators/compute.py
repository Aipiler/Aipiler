import operator


class ComputeOperator:

    def __init__(self, func, name: str):
        self.func = func  # The actual function for the operator
        self.name = name

    def __repr__(self):
        return self.name


# Predefined compute operators
ADD = ComputeOperator(operator.add, "add")
MUL = ComputeOperator(operator.mul, "mul")
SUB = ComputeOperator(operator.sub, "sub")
DIV = ComputeOperator(operator.floordiv, "div")
MIN = ComputeOperator(operator.lt, "min")
MAX = ComputeOperator(operator.ge, "max")


def pass_through(x):
    return x


PASS_THROUGH = ComputeOperator(
    pass_through, "PassThrough"
)  # For TopK/Gather like cases
