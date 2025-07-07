from typing import List, Union, Sequence, Optional, Set, Dict, Union, overload
import dataclasses
from enum import Enum, auto
import sys
import inspect
from abc import ABC, abstractmethod


class Dim:
    def __init__(self, size: Union[int, str]):
        self._size: Union[int, str]
        self._set_size(size)
        self._fake_tensor = None
        self._idx_in_tensor: int = None

    @property
    def is_dynamic(self):
        return self._is_dynamic

    @property
    def fake_tensor(self):
        return self._fake_tensor

    @property
    def index_in_tensor(self):
        return self._idx_in_tensor

    def _set_size(self, size: int | str):
        self._size = size
        if isinstance(size, str):
            self._is_dynamic = True
        elif isinstance(size, int):
            if size < 0:
                raise ValueError("Size cannot be negative.")
            self._is_dynamic = False
        else:
            raise TypeError(f"Expected int or str, got {type(size)}")

    @property
    def size(self) -> Optional[int | str]:
        """
        Get the size of the dimension.
        If the size is dynamic, raise an error.
        """
        return self._size

    def __repr__(self):
        from Aipiler.utils.namer import N

        return "{}(size={}, from={})".format(
            N.get_or_create_name_of(self),
            self.size,
            N.get_or_create_name_of(self._fake_tensor) if self._fake_tensor else "wild",
        )


def dim(size: int | str) -> Dim:
    assert isinstance(size, (int, str))
    d = Dim(size)
    return d


# 使用示例：
# create_dims(3, 4)           # 多个参数
# create_dims([3, 4])         # 序列参数
# create_dims("w", "h")       # 多个字符串参数
# create_dims(["w", "h"])     # 字符串序列参数


@overload
def dims(sizes: Sequence[Union[int, str]]) -> Sequence[Dim]: ...


@overload
def dims(*sizes: Union[int, str]) -> Sequence[Dim]: ...


def dims(*args) -> Sequence[Dim]:
    # 如果只有一个参数且是序列类型
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        sizes = args[0]
    else:
        sizes = args

    return [dim(s) for s in sizes]


class ValueDimSet:
    def __init__(self, dim_set: Optional[Set[Dim]] = None):
        self.dim_set: Optional[Set[Dim]] = dim_set if dim_set is not None else set()
        # if all dims in set are dynamic, self.value is None
        self.value: int = None

    def get_value(self) -> int:
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
        If self.value is not None
        """
        self.update_value()
        if self.value is None:
            return
        else:
            for dim in self.dim_set:
                dim._set_size(self.value)

    def update_value(self):
        """
        Update the value of the dimension set based on the dimensions it contains.
        If all dimensions have a fixed size, assert all size are equal, the value is fixed size.
        """
        self.value = None
        if not self.dim_set:
            return

        for dim in self.dim_set:
            if not dim.is_dynamic:
                size = dim.size
                assert isinstance(size, int)
                if self.value is not None and self.value != size:
                    raise ValueError(
                        f"Dimension size mismatch: {dim.size} != {self.value}"
                    )
                else:
                    self.value = size

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
        value_dim_set: ValueDimSet = None
        # find dim set
        for elem in elements:
            if elem in self.dim_set_dict:
                value_dim_set = self.dim_set_dict[elem]
                break

        if value_dim_set:  # if value dim exist, update dim set
            for elem in elements:
                value_dim_set.add(elem)
                if elem not in self.dim_set_dict:
                    self.dim_set_dict[elem] = value_dim_set
        else:  # else create new set and update it
            value_dim_set = ValueDimSet(set(elements))
            for elem in elements:
                self.dim_set_dict[elem] = value_dim_set

        # for element in elements:
        #     if not isinstance(element, Dim):
        #         raise TypeError(f"Expected Dim, got {type(element)}")
        #     if element in self.dim_set_dict:
        #         value_dim_set = self.dim_set_dict[element]

        #     element_set = self.find(element)
        #     if element_set is None:
        #         element_set = ValueDimSet({element})
        #     value_dim_set.union(element_set)
        #     for dim in element_set.dim_set:
        #         self.dim_set_dict[dim] = value_dim_set

    def is_connected(self, element1: Dim, element2: Dim) -> bool:
        if element1 not in self.dim_set_dict or element2 not in self.dim_set_dict:
            return False
        return self.dim_set_dict[element1] == self.dim_set_dict[element2]

    def get_all_value_dim_set(self) -> list[ValueDimSet]:
        return self.dim_set_dict.values()
