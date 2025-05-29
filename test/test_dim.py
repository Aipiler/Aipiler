from torch.export import Dim, export
import torch
from enum import Enum, auto
import sys
import inspect
from typing import List, Union, Sequence, Optional
import dataclasses


class _Dim(type):
    """
    Metaclass for :func:`Dim` types.
    """

    @staticmethod
    def readable(name, min_, max_):
        from torch.utils._sympy.numbers import int_oo

        if min_ == 2:
            min_ = None
        if max_ == int_oo:
            max_ = None
        if min_ is None and max_ is None:
            return f"Dim('{name}')"
        if min_ is None:
            return f"Dim('{name}', max={max_})"
        if max_ is None:
            return f"Dim('{name}', min={min_})"
        return f"Dim('{name}', min={min_}, max={max_})"

    def __add__(cls, other):
        # e.g., dim + 1
        if type(other) is not int:
            raise NotImplementedError(
                f"Attempted to add {other} to {cls.__name__}, where an integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return cls._derive(lambda x: x + other)

    def __radd__(cls, other):
        return cls + other

    def __sub__(cls, other):
        # e.g., dim - 1
        if type(other) is not int:
            raise NotImplementedError(
                f"Attempted to subtract {other} from {cls.__name__}, where an integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return cls._derive(lambda x: x - other)

    def __rsub__(cls, other):
        raise NotImplementedError(
            f"Attempted to negate {cls.__name__}. "
            "(Only increasing linear operations with integer coefficients are supported.)"
        )

    def __mul__(cls, other):
        # e.g., dim * 2
        if type(other) is not int or other <= 0:
            raise NotImplementedError(
                f"Attempted to multiply {other} with {cls.__name__}, where a positive integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return cls._derive(lambda x: x * other)

    def __rmul__(cls, other):
        return cls * other

    def create_alias(cls):
        """创建一个指向相同语义维度的新Dim对象 (别名)"""
        fn = lambda x: x  # identity function
        return _DerivedDim(cls._derived_name(fn), (int,), {"root": cls, "fn": fn})

    def _derived_name(cls, fn):
        from sympy import sympify

        return str(fn(sympify(cls.__name__)))

    def _derive(cls, fn):
        return _DerivedDim(cls._derived_name(fn), (int,), {"root": cls, "fn": fn})


class _DerivedDim(_Dim):
    """
    Metaclass for derived :func:`Dim` types.

    Currently we only support increasing linear expressions with integer coefficients.
    In other words, a derived Dim can always be written in the form Ax + B, where
    x is a regular Dim (i.e., non-derived Dim), A and B are integers, and A is positive.
    (In particular, the latter ensures that x < y => Ax + B < Ay + B.)
    These restrictions on the form of derived Dims makes the metatheory simpler: e.g.,
    it simplifies computing ranges for derived Dims, solving for underlying regular Dims,
    deciding equalities between derived Dims, and so on.

    The function lambda x: Ax + B is expressed by `fn`, where x is a normal Dim, `root`.
    The range of a derived Dim is computed by mapping `fn` over the range of its `root`.
    """

    @property
    def min(self):
        # assume that self.fn is an increasing function
        # TODO(avik): use sympy value range analysis instead?
        from sympy import Integer

        from torch.utils._sympy.numbers import int_oo

        if self.root.min is -int_oo:  # type: ignore[attr-defined]
            return -int_oo  # fn not needed cuz increasing

        _min_symint = self.fn(Integer(self.root.min))  # type: ignore[attr-defined]
        root = self.root  # type: ignore[attr-defined]
        assert _min_symint >= 0, (
            f"Expected derived min value of {self.__name__} to be >= 0. "
            f"Please specify an appropriate min value for {root.__name__} "
            f"(currently {root.min})."
        )
        return int(_min_symint)

    @property
    def max(self):
        # assume that self.fn is an increasing function
        # TODO(avik): use sympy value range analysis instead?
        from sympy import Integer

        from torch.utils._sympy.numbers import int_oo

        if self.root.max is int_oo:  # type: ignore[attr-defined]
            return int_oo  # fn not needed cuz increasing

        _max_symint = self.fn(Integer(self.root.max))  # type: ignore[attr-defined]
        root = self.root  # type: ignore[attr-defined]
        assert _max_symint <= sys.maxsize - 1, (
            f"Expected derived max value of {self.__name__} to be <= {sys.maxsize - 1}. "
            f"Please specify an appropriate max value for {root.__name__} "
            f"(currently {root.max})."
        )
        return int(_max_symint)

    def _derive(self, fn):
        # We support nesting, e.g., 2*dim + 1.
        # This is implemented by composing operations on the same root.
        # As a consequence, roots are always regular Dims (i.e., not derived Dims).
        return _DerivedDim(
            self._derived_name(fn),
            (int,),
            {"root": self.root, "fn": lambda x: fn(self.fn(x))},  # type: ignore[attr-defined]
        )


def Dim(name: str, *, min: Optional[int] = None, max: Optional[int] = None):
    """
    :func:`Dim` constructs a type analogous to a named symbolic integer with a range.
    It can be used to describe multiple possible values of a dynamic tensor dimension.
    Note that different dynamic dimensions of the same tensor, or of different tensors,
    can be described by the same type.

    Args:
        name (str): Human-readable name for debugging.
        min (Optional[int]): Minimum possible value of given symbol (inclusive)
        max (Optional[int]): Maximum possible value of given symbol (inclusive)

    Returns:
        A type that can be used in dynamic shape specifications for tensors.
    """

    from torch.utils._sympy.numbers import int_oo

    _min = 0 if min is None else min
    _max = int_oo if max is None else max
    assert _max > _min, f"Cannot create Dim with inconsistent min={min}, max={max}"
    assert name.isidentifier(), f"Dim name must be a valid identifier, got {name}"
    dim = _Dim(name, (int,), {"min": _min, "max": _max})
    dim.__module__ = getattr(
        inspect.getmodule(inspect.stack()[1][0]), "__name__", "__main__"
    )
    return dim


def test_dim():
    d1 = Dim("d1", min=0, max=10)
    d2 = Dim("d2", min=5, max=15)
    d3 = d1 + 2
    d4 = 3 * d2
    d5 = d1.create_alias()

    assert d1.min == 0
    assert d1.max == 10
    assert d3.min == 2
    assert d3.max == 12
    assert d4.min == 15
    assert d4.max == 45
    assert d5.min == 0
    assert d5.max == 10

    print("All tests passed!")
