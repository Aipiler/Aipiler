from typing import List, Union, Sequence, Optional, Set, Dict
import dataclasses
from enum import Enum, auto
import sys
import inspect
from abc import ABC, abstractmethod


class Dim:
    def __init__(self):
        self.fake_tensor = None
        self.idx = None
        self.size: int | str = None
        self.is_dynamic: bool = False

    def set_fake_tensor(self, fake_tensor, idx: int):
        from Aipiler.tensor import FakeTensor

        if not isinstance(fake_tensor, FakeTensor):
            raise TypeError(f"Expected FakeTensor, got {type(fake_tensor)}")
        self.fake_tensor = fake_tensor
        self.idx = idx

    def set_size(self, size: int | str):
        self.size = size
        if isinstance(size, str):
            self.is_dynamic = True
        elif isinstance(size, int):
            if size < 0:
                raise ValueError("Size cannot be negative.")
            self.is_dynamic = False
        else:
            raise TypeError(f"Expected int or str, got {type(size)}")

    def get_size(self) -> Optional[int | str]:
        """
        Get the size of the dimension.
        If the size is dynamic, raise an error.
        """
        return self.size


class ValueDimSet:
    def __init__(self, dim_set: Optional[Set[Dim]] = None):
        self.dim_set: Optional[Set[Dim]] = dim_set if dim_set is not None else set()
        self.value: int | str = None

    def get_value(self) -> int | str:
        """
        Get the value of the dimension set.
        If all dimensions have a fixed size, return the product of those sizes.
        If any dimension is dynamic, return None.
        """
        if self.value is None:
            self.update_value()
        return self.value

    def populate_dim_size(self):
        """
        Populate the sizes of the dimensions in the set.
        If any dimension is dynamic, raise an error.
        """
        self.update_value()
        for dim in self.dim_set:
            if dim.size is None:
                dim.set_size(self.value)
            else:
                if dim.size != self.value:
                    raise ValueError(
                        f"Dimension size mismatch: {dim.size} != {self.value}"
                    )

    def update_value(self):
        """
        Update the value of the dimension set based on the dimensions it contains.
        If all dimensions have a fixed size, the value is the product of those sizes.
        If any dimension is dynamic, the value is None.
        """
        if not self.dim_set:
            self.value = None
            return

        for dim in self.dim_set:
            if dim.get_size() is not None:
                self.value = dim.size
                return

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

    def get_all_value_dim_set(self) -> list[ValueDimSet]:
        return self.dim_set_dict.values()
