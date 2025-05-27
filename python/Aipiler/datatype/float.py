from typing import Any
from functools import cached_property
from dataclasses import dataclass
import warnings
import numpy as np
from Aipiler.datatype import DataType


@dataclass
class FloatInfo:
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: DataType


class FloatType(DataType):
    def __init__(self, name, short_name, nbytes, min_value, max_value, eps, smallest_normal):
        super().__init__(name, short_name, nbytes)

        self._min_value: float = min_value
        self._max_value: float = max_value
        self._eps: float = eps
        self._smallest_normal: float = smallest_normal

    def is_integer_subbyte(self) -> bool:
        return False

    def is_float(self) -> bool:
        return True

    def is_integer(self) -> bool:
        return False

    def is_complex(self) -> bool:
        return False

    def is_vector(self) -> bool:
        return False

    def is_any_float16(self):
        return self.short_name in ['f16', 'bf16']

    def is_boolean(self) -> bool:
        return False

    def finfo(self) -> FloatInfo:
        return FloatInfo(
            bits=self.nbytes * 8,
            eps=self._eps,
            max=self._max_value,
            min=self._min_value,
            smallest_normal=self._smallest_normal,
            dtype=self,
        )


float16 = FloatType(
    'float16',
    'f16',
    2,
    np.finfo(np.float16).min,
    np.finfo(np.float16).max,
    np.finfo(np.float16).eps,
    np.finfo(np.float16).tiny,
)
float32 = FloatType(
    'float32',
    'f32',
    4,
    np.finfo(np.float32).min,
    np.finfo(np.float32).max,
    np.finfo(np.float32).eps,
    np.finfo(np.float32).tiny,
)
float64 = FloatType(
    'float64',
    'f64',
    8,
    np.finfo(np.float64).min,
    np.finfo(np.float64).max,
    np.finfo(np.float64).eps,
    np.finfo(np.float64).tiny,
)
bfloat16 = FloatType('bfloat16', 'bf16', 2, -3.4e38, 3.4e38, None, None)
tfloat32 = FloatType('tfloat32', 'tf32', 4, -3.4e38, 3.4e38, None, None)

f16 = float16
f32 = float32
f64 = float64
bf16 = bfloat16
tf32 = tfloat32
