import inspect
from typing import List, Optional, Union, Callable, Dict, Any, Tuple
from enum import Enum, auto
from abc import ABC, abstractmethod
from .range import Range, CompoundRange
from collections import defaultdict
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .einsumExpression import EinsumExpression


class Dtype(Enum):
    """Represents the data type of a tensor."""

    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    # Add more types as needed


# TODO: Empty Value
class Empty(ABC):
    pass


class ZeroEmpty(Empty):
    def __init__(self):
        pass

    def __repr__(self):
        return "0"


# TODO: 暂时不考虑symbolic shape的情况，暂时考虑确定shape。
class Rank:
    def __init__(self, size: int):
        self.data_space = None
        self.size = size
        self.rankSet = []
        self.use_list = []
        self.dependent_rank = []

    def belong_to(self, rank_set: "RankSet"):
        self.rankSet.append(rank_set)

    # def init_from_other_rank(self, other: "Rank", size: int = 0):
    #     self.size = other.size
    #     self.def_rank = other
    #     other.add_use_rank(self)

    def add_use_rank(self, use: "Rank"):
        if use in self.use_list:
            raise KeyError("use当前rank的other rank已经被加入use list，存在错误")
        self.use_list.append(use)

    def set_data_space(self, data_space: "DataSpace"):
        self.data_space = data_space

    def set_name(self, name: str):
        """Set the name of the tensor rank."""
        self.name = name

    def get_size(self) -> int:
        """Get the range of the tensor rank."""
        return self.size


class Tensor:
    def __init__(self):
        self.shape = ()

        # 由shape创建rank
        self.ranks: List[Rank]

    def from_torch_tensor(self, torch_tensor: torch.Tensor):
        self.shape = torch_tensor.shape
        self.dtype = torch_tensor.dtype


class RankSet:
    def __init__(self, ranks: List[Rank]):
        """Initialize a set of ranks."""
        self.ranks: List[Rank] = []  # rank是存在顺序的
        for rank in ranks:
            if isinstance(rank, Rank):
                self.add_rank(rank)
            else:
                raise TypeError("Rank must be an instance of Rank.")

        self.rank = len(ranks)
        self.shape = () if self.rank == 0 else (rank.get_size() for rank in ranks)
        self.data_space = None

    def set_data_space(self, data_space: "DataSpace"):
        self.data_space = data_space

    def add_rank(self, rank: Rank):
        """Add a rank to the rank set."""
        rank.belong_to(self)
        self.ranks.append(rank)

    def get_volume(self) -> int:
        """Get the volume of the tensor rank set."""
        volume = 1
        for rank in self.ranks:
            volume *= rank.get_size()
        return volume

    # def gen_data_space(self, dtype: Dtype, empty: Empty) -> "DataSpace":
    #     """Generate a data space from the rank set."""
    #     data_space = DataSpace(self, dtype, empty)
    #     self.set_data_space(data_space)
    #     return data_space

    def __getitem__(self, key: int) -> Rank | None:
        if key < self.rank:
            return self.ranks[key]
        else:
            return None

    def __iter__(self):
        self.index = 0  # 记录当前索引
        return self  # 返回迭代器对象

    def __next__(self):
        if self.index < self.rank:
            result = self.ranks[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration  # 当没有更多元素时，抛出 StopIteration


class DataSpace:

    def __init__(
        self,
        einsum: EinsumExpression,
        rankset: RankSet,
        dtype: Dtype = Dtype.FLOAT,
        emptyType: Empty = ZeroEmpty(),
    ):
        """Initialize a data space with a name."""
        self.default_build_rank_set = rankset
        self.einsum = einsum
        self.volume = rankset.get_volume()
        self.dtype = dtype
        self.emptyType = emptyType
        self.dim_list = defaultdict(set)  # {0: {rank1, rank2}, 1: {rank3}, 2: {}, ...}
        self.rank_set_list = [rankset]
        for pos, rank in enumerate(rankset):
            self.dim_list[pos] = rank

    # def init_from_other_rank_set(self, rank_set: RankSet) -> RankSet:
    #     self.default_build_rank_set = rank_set
    #     self.volume = rank_set.get_volume()
    #     ranks = []
    #     for pos, rank in enumerate(rank_set):
    #         new_rank = Rank()
    #         new_rank.init_from_other_rank(rank)
    #         ranks.append(new_rank)
    #         self.dim_list[pos].add(new_rank)

    #     new_rank_set = RankSet(ranks)
    #     self.rank_set_list.append(new_rank_set)
    #     return new_rank_set

    def find_rank_from_pos(self, pos: int, size: int) -> Optional[Rank]:
        """Find a rank from the position and size."""
        if pos in self.dim_list:
            for rank in self.dim_list[pos]:
                if rank.get_size() == size:
                    return rank
        return None

    def gen_rank_set_from_shape(self, shape: tuple[int]) -> RankSet:
        """Generate a new rank set with the specified shape."""
        volume = 1
        for size in shape:
            volume *= size
        if volume != self.volume:
            raise ValueError("self.volume != shape volume")
        if self.volume == 0:
            self.volume = volume
        ranks = []
        for i, size in enumerate(shape):
            rank = self.find_rank_from_pos(i, size)
            if rank is None:
                rank = Rank(size=size)  # def 新的rank
                rank.set_data_space(self)  # 将新的rank设置所属的data space
                self.dim_list[i].add(rank)  # 将新的rank纳入dataspace的管理
            ranks.append(rank)
        new_rank_set = RankSet(ranks)  # 创建新的RankSet
        new_rank_set.set_data_space(self)  # 设置rankset的data space
        self.rank_set_list.append(new_rank_set)  # 将rank set纳入data space的管理
        return new_rank_set
