from functools import cached_property
from Aipiler.datatype import DataType


class Boolean(DataType):
    def __init__(self):
        super().__init__('bool', 'bool', 1)

    def is_integer_subbyte(self) -> bool:
        return False

    def is_float(self) -> bool:
        return False

    def is_integer(self) -> bool:
        return True

    def is_complex(self) -> bool:
        return False

    def is_vector(self) -> bool:
        return False

    def is_boolean(self) -> bool:
        return True

    @cached_property
    def one(self):
        return self.constant(True)

    @cached_property
    def zero(self):
        return self.constant(False)

    @cached_property
    def true(self):
        return self.constant(True)

    @cached_property
    def false(self):
        return self.constant(False)

    @property
    def min_value(self):
        raise ValueError('Boolean type has no minimum value.')

    @property
    def max_value(self):
        raise ValueError('Boolean type has no maximum value.')


boolean = Boolean()
