class UnaryOperator:

    def __init__(self, func, name: str):
        self.func = func  # The actual function for the operator
        self.name = name

    def __repr__(self):
        return self.name


# Predefined unary operators


def relu(x):
    return x if x > 0 else 0


RELU = UnaryOperator(relu, "relu")
# ... add others as needed
