import __future__


class DataType:
    """
    The data type that defines how to interpret the data in memory.

    """

    def __init__(self, name: str, short_name: str, nbytes: int):
        self._name: str = name
        self._short_name: str = short_name
        self._nbytes: int = nbytes

    def __str__(self):
        return 'aipiler.{}'.format(self.name)

    def __eq__(self, other):
        return isinstance(other, DataType) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def short_name(self) -> str:
        return self._short_name

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def nbits(self) -> int:
        return self._nbytes * 8


    def is_integer_subbyte(self) -> bool:
        raise NotImplementedError()

    def is_float(self) -> bool:
        raise NotImplementedError()

    def is_integer(self) -> bool:
        raise NotImplementedError()

    def is_complex(self) -> bool:
        raise NotImplementedError()

    def is_vector(self) -> bool:
        raise NotImplementedError()

    def is_boolean(self) -> bool:
        raise NotImplementedError()

    def is_any_float16(self) -> bool:
        raise NotImplementedError()
