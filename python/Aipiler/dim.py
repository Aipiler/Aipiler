from typing import List, Union, Sequence, Optional, Set, Dict
import dataclasses
from enum import Enum, auto
import sys
import inspect
from abc import ABC, abstractmethod


@dataclasses.dataclass
class SymIntArgument:
    name: str


@dataclasses.dataclass
class SymFloatArgument:
    name: str


@dataclasses.dataclass
class SymBoolArgument:
    name: str


class Dim:
    def __init__(self, fake_tensor):
        from Aipiler.tensor import FakeTensor

        self.fake_tensor = fake_tensor


class ValueDimSet:
    def __init__(self, dim_set: Optional[Set] = None, value: Optional[int] = None):
        self.dim_set = dim_set if dim_set is not None else set()
        self.value = value

    def add(self, dim: Dim):

        if isinstance(dim, Dim):
            self.dim_set.add(dim)
        else:
            raise TypeError(f"Expected Dim, got {type(dim)}")

    def find(self, dim: Dim) -> bool:
        if isinstance(dim, Dim):
            return dim in self.dim_set
        else:
            raise TypeError(f"Expected Dim, got {type(dim)}")

    def union(self, other: "ValueDimSet") -> "ValueDimSet":
        if isinstance(other, ValueDimSet):
            return ValueDimSet(
                dim_set=self.dim_set | other.dim_set,
                value=self.value if self.value is not None else other.value,
            )
        else:
            raise TypeError(f"Expected DimSet, got {type(other)}")


class DisjointSetUnion:

    def __init__(
        self,
    ):
        self.dim_set_dict: dict[Dim, ValueDimSet] = {}

    def find(self, element: Dim) -> Optional[ValueDimSet]:
        return self.dim_set_dict.get(element, None)

    def union(self, *elements: Dim):
        if len(elements) < 2:
            raise ValueError("At least two elements are required to perform union.")
        value_dim_set = ValueDimSet()
        for element in elements:
            if not isinstance(element, Dim):
                raise TypeError(f"Expected Dim, got {type(element)}")
            if element not in self.dim_set_dict:
                self.dim_set_dict[element] = ValueDimSet()
            value_dim_set = value_dim_set.union(self.dim_set_dict[element])
            self.dim_set_dict[element] = value_dim_set

    def is_connected(self, element1: Dim, element2: Dim) -> bool:
        if element1 not in self.dim_set_dict or element2 not in self.dim_set_dict:
            raise ValueError(
                f"One or both elements {element1}, {element2} are not in the disjoint set."
            )
        return self.dim_set_dict[element1] == self.dim_set_dict[element2]
