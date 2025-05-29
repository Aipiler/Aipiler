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
    def __init__(
        self,
    ):
        self.fake_tensor = None
        self.idx = None

    def set_fake_tensor(self, fake_tensor, idx: int):
        from Aipiler.tensor import FakeTensor

        if not isinstance(fake_tensor, FakeTensor):
            raise TypeError(f"Expected FakeTensor, got {type(fake_tensor)}")
        self.fake_tensor = fake_tensor
        self.idx = idx

    def get_fake_tensor(self):
        if self.fake_tensor is None:
            raise ValueError("Fake tensor is not set.")
        return self.fake_tensor

    def get_index(self) -> int:
        if self.idx is None:
            raise ValueError("Index is not set.")
        return self.idx


class ValueDimSet:
    def __init__(self, dim_set: Optional[Set[Dim]] = None, value: Optional[int] = None):
        self.dim_set: Optional[Set[Dim]] = dim_set if dim_set is not None else set()
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

    def union(self, other: "ValueDimSet"):
        if not isinstance(other, ValueDimSet):
            raise TypeError(f"Expected ValueDimSet, got {type(other)}")
        self.dim_set |= other.dim_set


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
            element_set = self.find(element)
            if element_set is None:
                element_set = ValueDimSet({element})
            value_dim_set.union(element_set)
            for dim in element_set.dim_set:
                self.dim_set_dict[dim] = value_dim_set

    def is_connected(self, element1: Dim, element2: Dim) -> bool:
        if element1 not in self.dim_set_dict or element2 not in self.dim_set_dict:
            raise ValueError(
                f"One or both elements {element1}, {element2} are not in the disjoint set."
            )
        return self.dim_set_dict[element1] == self.dim_set_dict[element2]
