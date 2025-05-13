from typing import List, Tuple, Dict, Any, Optional, Union, TypeVar, Set
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass


class MemoryLevel(Enum):
    """定义内存层次结构"""

    EXTERNAL = auto()  # 外部存储（如磁盘、网络存储）
    DRAM = auto()  # 主内存
    L3_CACHE = auto()  # 三级缓存
    L2_CACHE = auto()  # 二级缓存
    L1_CACHE = auto()  # 一级缓存
    REGISTER = auto()  # 寄存器
    SHARED_MEMORY = auto()  # 共享内存（GPU）
    LOCAL_MEMORY = auto()  # 本地内存（GPU）
    CONSTANT_MEMORY = auto()  # 常量内存（GPU）
    TEXTURE_MEMORY = auto()  # 纹理内存（GPU）


class DataLayout(Enum):
    """定义数据布局格式"""

    NCHW = auto()  # 批次, 通道, 高度, 宽度 (常见于深度学习)
    NHWC = auto()  # 批次, 高度, 宽度, 通道 (TensorFlow常用)
    CHWN = auto()  # 通道, 高度, 宽度, 批次
    CHW = auto()  # 通道, 高度, 宽度 (无批次)
    HWC = auto()  # 高度, 宽度, 通道 (图像常用)
    NC = auto()  # 批次, 通道 (全连接层常用)
    CUSTOM = auto()  # 自定义布局


class PrecisionType(Enum):
    """定义精度类型"""

    FP32 = auto()  # 32位浮点数
    FP16 = auto()  # 16位浮点数
    BF16 = auto()  # Brain浮点数
    INT32 = auto()  # 32位整数
    INT16 = auto()  # 16位整数
    INT8 = auto()  # 8位整数
    UINT8 = auto()  # 无符号8位整数
    QUINT8 = auto()  # 量化的无符号8位整数
    QINT8 = auto()  # 量化的8位整数
    QUINT4 = auto()  # 量化的无符号4位整数
    QINT4 = auto()  # 量化的4位整数
    BOOL = auto()  # 布尔类型
    CUSTOM = auto()  # 自定义精度


@dataclass
class QuantizationParams:
    """量化参数"""

    scale: float = 1.0  # 缩放因子
    zero_point: int = 0  # 零点
    min_value: float = 0.0  # 最小值
    max_value: float = 0.0  # 最大值
    axis: Optional[int] = None  # 量化轴（用于按通道量化）
    num_bits: int = 8  # 位宽
    is_signed: bool = True  # 是否有符号


class Data:
    """数据视图基类 - 表示计算图中流动的数据"""

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: Union[str, PrecisionType, np.dtype] = PrecisionType.FP32,
        name: Optional[str] = None,
        memory_level: MemoryLevel = MemoryLevel.DRAM,
        layout: Optional[DataLayout] = None,
        metadata: Optional[Dict[str, Any]] = None,
        alignment: int = 1,
    ):
        """
        初始化数据对象

        Args:
            shape: 数据形状，如(1, 3, 224, 224)
            dtype: 数据类型，可以是字符串、PrecisionType枚举或np.dtype
            name: 数据名称
            memory_level: 存储级别
            layout: 数据布局
            metadata: 元数据
            alignment: 内存对齐字节数
        """
        self.shape = shape
        self._set_dtype(dtype)
        self.name = name or f"data_{id(self)}"
        self.memory_level = memory_level
        self.layout = layout
        self.metadata = metadata or {}
        self.alignment = alignment
        self.quantization = None  # 量化参数，默认为None（非量化）
        self.is_constant = False  # 是否为常量数据
        self.is_view = False  # 是否为其他数据的视图
        self.consumers = set()  # 使用此数据的节点
        self.producer = None  # 产生此数据的节点
        self._value = None  # 实际数据值（延迟计算）

    def _set_dtype(self, dtype: Union[str, PrecisionType, np.dtype]) -> None:
        """设置数据类型，处理不同的输入格式"""
        if isinstance(dtype, PrecisionType):
            self.dtype = dtype
            # 映射到numpy数据类型
            dtype_map = {
                PrecisionType.FP32: np.float32,
                PrecisionType.FP16: np.float16,
                PrecisionType.INT32: np.int32,
                PrecisionType.INT16: np.int16,
                PrecisionType.INT8: np.int8,
                PrecisionType.UINT8: np.uint8,
                PrecisionType.BOOL: np.bool_,
            }
            self.np_dtype = dtype_map.get(dtype, np.float32)
        elif isinstance(dtype, str):
            # 从字符串解析数据类型
            try:
                self.np_dtype = np.dtype(dtype)
                # 反向映射到PrecisionType
                if np.issubdtype(self.np_dtype, np.floating):
                    if self.np_dtype.itemsize == 4:
                        self.dtype = PrecisionType.FP32
                    elif self.np_dtype.itemsize == 2:
                        self.dtype = PrecisionType.FP16
                elif np.issubdtype(self.np_dtype, np.integer):
                    if self.np_dtype.itemsize == 4:
                        self.dtype = PrecisionType.INT32
                    elif self.np_dtype.itemsize == 2:
                        self.dtype = PrecisionType.INT16
                    elif self.np_dtype.itemsize == 1:
                        if np.issubdtype(self.np_dtype, np.signedinteger):
                            self.dtype = PrecisionType.INT8
                        else:
                            self.dtype = PrecisionType.UINT8
                elif np.issubdtype(self.np_dtype, np.bool_):
                    self.dtype = PrecisionType.BOOL
                else:
                    self.dtype = PrecisionType.CUSTOM
            except TypeError:
                # 无法解析字符串，使用默认值
                self.dtype = PrecisionType.CUSTOM
                self.np_dtype = np.float32
        elif isinstance(dtype, np.dtype):
            self.np_dtype = dtype
            # 映射到PrecisionType（同上）
            if np.issubdtype(dtype, np.floating):
                if dtype.itemsize == 4:
                    self.dtype = PrecisionType.FP32
                elif dtype.itemsize == 2:
                    self.dtype = PrecisionType.FP16
            elif np.issubdtype(dtype, np.integer):
                if dtype.itemsize == 4:
                    self.dtype = PrecisionType.INT32
                elif dtype.itemsize == 2:
                    self.dtype = PrecisionType.INT16
                elif dtype.itemsize == 1:
                    if np.issubdtype(dtype, np.signedinteger):
                        self.dtype = PrecisionType.INT8
                    else:
                        self.dtype = PrecisionType.UINT8
            elif np.issubdtype(dtype, np.bool_):
                self.dtype = PrecisionType.BOOL
            else:
                self.dtype = PrecisionType.CUSTOM
        else:
            # 未知类型，使用默认值
            self.dtype = PrecisionType.FP32
            self.np_dtype = np.float32

    @property
    def ndim(self) -> int:
        """获取维度数量"""
        return len(self.shape)

    @property
    def size(self) -> int:
        """获取元素总数"""
        return np.prod(self.shape)

    @property
    def itemsize(self) -> int:
        """获取每个元素的字节数"""
        return np.dtype(self.np_dtype).itemsize

    @property
    def nbytes(self) -> int:
        """获取总字节数"""
        return self.size * self.itemsize

    def set_value(self, value: Any) -> None:
        """设置实际数据值"""
        self._value = value

    def get_value(self) -> Any:
        """获取实际数据值"""
        return self._value

    def has_value(self) -> bool:
        """检查是否有实际数据值"""
        return self._value is not None

    def set_quantization(self, params: QuantizationParams) -> None:
        """设置量化参数"""
        self.quantization = params

    def is_quantized(self) -> bool:
        """检查数据是否已量化"""
        return self.quantization is not None

    def set_producer(self, node) -> None:
        """设置产生此数据的节点"""
        self.producer = node

    def add_consumer(self, node) -> None:
        """添加使用此数据的节点"""
        self.consumers.add(node)

    def remove_consumer(self, node) -> None:
        """移除使用此数据的节点"""
        if node in self.consumers:
            self.consumers.remove(node)

    def set_constant(self, is_constant: bool = True) -> None:
        """设置数据是否为常量"""
        self.is_constant = is_constant

    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据"""
        return self.metadata.get(key, default)

    def __str__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}(name={self.name}, "
            f"shape={self.shape}, dtype={self.dtype}, "
            f"memory_level={self.memory_level.name})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": (
                self.dtype.name if isinstance(self.dtype, Enum) else str(self.dtype)
            ),
            "memory_level": self.memory_level.name,
            "layout": self.layout.name if self.layout else None,
            "is_constant": self.is_constant,
            "is_view": self.is_view,
            "quantized": self.is_quantized(),
            "nbytes": self.nbytes,
            "metadata": self.metadata,
        }


class Tensor(Data):
    """张量视图 - 完整的多维数据"""

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: Union[str, PrecisionType, np.dtype] = PrecisionType.FP32,
        name: Optional[str] = None,
        layout: DataLayout = DataLayout.NCHW,
        memory_level: MemoryLevel = MemoryLevel.DRAM,
        strides: Optional[Tuple[int, ...]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        alignment: int = 16,  # 默认16字节对齐
    ):
        """
        初始化张量

        Args:
            shape: 张量形状
            dtype: 数据类型
            name: 张量名称
            layout: 数据布局
            memory_level: 存储级别
            strides: 步长（每个维度上相邻元素之间的字节数）
            metadata: 元数据
            alignment: 内存对齐字节数
        """
        super().__init__(shape, dtype, name, memory_level, layout, metadata, alignment)

        # 计算默认步长（如果未提供）
        if strides is None:
            self.strides = self._compute_default_strides()
        else:
            self.strides = strides

        # 张量特有属性
        self.is_contiguous = self._check_contiguous()
        self.requires_grad = False  # 是否需要梯度
        self.grad = None  # 梯度

    def _compute_default_strides(self) -> Tuple[int, ...]:
        """计算默认步长"""
        itemsize = self.itemsize
        strides = [itemsize]
        for dim in reversed(self.shape[:-1]):
            strides.append(strides[-1] * dim)
        return tuple(reversed(strides))

    def _check_contiguous(self) -> bool:
        """检查张量是否连续存储"""
        expected_strides = self._compute_default_strides()
        return self.strides == expected_strides

    def transpose(self, *axes) -> "Tensor":
        """转置张量（创建视图）"""
        if not axes:
            axes = tuple(range(len(self.shape) - 1, -1, -1))
        elif len(axes) != len(self.shape):
            raise ValueError(f"转置轴数量 {len(axes)} 不匹配张量维度 {len(self.shape)}")

        # 计算新形状和步长
        new_shape = tuple(self.shape[i] for i in axes)
        new_strides = tuple(self.strides[i] for i in axes)

        # 创建新的张量视图
        result = Tensor(
            shape=new_shape,
            dtype=self.dtype,
            name=f"{self.name}_transposed",
            layout=DataLayout.CUSTOM,  # 转置后布局通常会改变
            memory_level=self.memory_level,
            strides=new_strides,
            metadata=self.metadata.copy(),
        )
        result.is_view = True
        result.set_value(self.get_value())  # 共享数据
        return result

    def reshape(self, *new_shape) -> "Tensor":
        """重塑张量（可能创建副本）"""
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = new_shape[0]

        # 处理-1维度
        shape_list = list(new_shape)
        neg_one_idx = None
        for i, dim in enumerate(shape_list):
            if dim == -1:
                if neg_one_idx is not None:
                    raise ValueError("最多只能有一个-1维度")
                neg_one_idx = i

        if neg_one_idx is not None:
            # 计算-1对应的实际维度
            size = np.prod(self.shape)
            other_dims_prod = -np.prod(shape_list)
            if size % other_dims_prod != 0:
                raise ValueError(f"无法将大小为{size}的张量重塑为{new_shape}")
            shape_list[neg_one_idx] = size // other_dims_prod

        final_shape = tuple(shape_list)

        # 检查元素总数是否匹配
        if np.prod(final_shape) != np.prod(self.shape):
            raise ValueError(f"无法将形状{self.shape}重塑为{final_shape}")

        # 如果张量是连续的，可以创建视图
        if self.is_contiguous:
            result = Tensor(
                shape=final_shape,
                dtype=self.dtype,
                name=f"{self.name}_reshaped",
                layout=DataLayout.CUSTOM,
                memory_level=self.memory_level,
                strides=None,  # 重新计算步长
                metadata=self.metadata.copy(),
            )
            result.is_view = True
            result.set_value(self.get_value())  # 共享数据
        else:
            # 非连续张量需要创建副本
            result = Tensor(
                shape=final_shape,
                dtype=self.dtype,
                name=f"{self.name}_reshaped_copy",
                layout=DataLayout.CUSTOM,
                memory_level=self.memory_level,
                metadata=self.metadata.copy(),
            )
            # 实际数据需要重新排列，这里仅是示例框架
            result.is_view = False
            # result.set_value(...)  # 需要实际重新排列数据

        return result

    def to(self, memory_level: MemoryLevel) -> "Tensor":
        """移动张量到指定存储级别"""
        # 创建新的张量对象
        result = Tensor(
            shape=self.shape,
            dtype=self.dtype,
            name=self.name,
            layout=self.layout,
            memory_level=memory_level,
            strides=self.strides,
            metadata=self.metadata.copy(),
            alignment=self.alignment,
        )

        # 如果有值，需要移动数据（实际实现可能更复杂）
        if self.has_value():
            result.set_value(self.get_value())

        return result


class Tile(Data):
    """数据块视图 - 用于缓存优化的数据切片"""

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: Union[str, PrecisionType, np.dtype],
        origin_tensor: Tensor,
        offsets: Tuple[int, ...],
        name: Optional[str] = None,
        memory_level: MemoryLevel = MemoryLevel.L2_CACHE,
        metadata: Optional[Dict[str, Any]] = None,
        alignment: int = 32,  # 通常L2缓存对齐要求更高
    ):
        """
        初始化数据块

        Args:
            shape: 数据块形状
            dtype: 数据类型
            origin_tensor: 源张量
            offsets: 在源张量中的偏移（每个维度）
            name: 数据块名称
            memory_level: 存储级别
            metadata: 元数据
            alignment: 内存对齐字节数
        """
        super().__init__(
            shape,
            dtype,
            name or f"tile_from_{origin_tensor.name}",
            memory_level,
            origin_tensor.layout,
            metadata,
            alignment,
        )

        self.origin = origin_tensor
        self.offsets = offsets
        self.is_view = True

        # 验证数据块在源张量范围内
        self._validate_bounds()

        # 计算内存占用和步长
        self.strides = origin_tensor.strides

        # 特有属性
        self.prefetch_hint = False  # 预取提示
        self.cache_hint = False  # 缓存提示

    def _validate_bounds(self) -> None:
        """验证数据块在源张量边界内"""
        if len(self.shape) != len(self.origin.shape):
            raise ValueError(
                f"数据块维度 {len(self.shape)} 与源张量维度 {len(self.origin.shape)} 不匹配"
            )

        if len(self.offsets) != len(self.shape):
            raise ValueError(
                f"偏移数量 {len(self.offsets)} 与形状维度 {len(self.shape)} 不匹配"
            )

        for i, (offset, tile_dim, tensor_dim) in enumerate(
            zip(self.offsets, self.shape, self.origin.shape)
        ):
            if offset < 0:
                raise ValueError(f"维度 {i} 的偏移 {offset} 为负")

            if offset + tile_dim > tensor_dim:
                raise ValueError(
                    f"维度 {i} 的数据块 ({offset}:{offset+tile_dim}) 超出源张量范围 (0:{tensor_dim})"
                )

    def get_absolute_position(
        self, relative_indices: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """计算相对于源张量的绝对位置"""
        if len(relative_indices) != len(self.offsets):
            raise ValueError(
                f"索引维度 {len(relative_indices)} 不匹配偏移维度 {len(self.offsets)}"
            )

        return tuple(
            offset + idx for offset, idx in zip(self.offsets, relative_indices)
        )

    def get_element(self, *indices) -> Any:
        """获取特定元素（需要源张量有数据）"""
        if not self.origin.has_value():
            raise ValueError("源张量没有数据，无法获取元素")

        absolute_indices = self.get_absolute_position(indices)
        origin_value = self.origin.get_value()

        # 假设origin_value支持索引操作
        return origin_value[absolute_indices]

    def set_prefetch_hint(self, hint: bool = True) -> None:
        """设置预取提示"""
        self.prefetch_hint = hint

    def set_cache_hint(self, hint: bool = True) -> None:
        """设置缓存提示"""
        self.cache_hint = hint


class Vector(Data):
    """向量视图 - 向量化执行单元"""

    def __init__(
        self,
        length: int,
        dtype: Union[str, PrecisionType, np.dtype],
        origin_tile: Union[Tile, Tensor],
        offset: Tuple[int, ...],
        axis: int = 0,
        name: Optional[str] = None,
        memory_level: MemoryLevel = MemoryLevel.REGISTER,
        metadata: Optional[Dict[str, Any]] = None,
        vector_width: Optional[int] = None,
        alignment: int = 64,  # 向量通常需要更高对齐
    ):
        """
        初始化向量

        Args:
            length: 向量长度
            dtype: 数据类型
            origin_tile: 源数据块或张量
            offset: 在源数据中的偏移
            axis: 向量化维度
            name: 向量名称
            memory_level: 存储级别
            metadata: 元数据
            vector_width: 向量宽度（SIMD/向量寄存器宽度）
            alignment: 内存对齐字节数
        """
        shape = [1] * len(origin_tile.shape)
        shape[axis] = length
        shape = tuple(shape)

        super().__init__(
            shape,
            dtype,
            name or f"vector_from_{origin_tile.name}",
            memory_level,
            None,  # 向量通常没有布局概念
            metadata,
            alignment,
        )

        self.origin = origin_tile
        self.offset = offset
        self.axis = axis
        self.is_view = True
        self.vector_width = vector_width or length

        # 验证向量在源数据范围内
        self._validate_bounds()

        # 特有属性
        self.unroll_hint = False  # 循环展开提示
        self.vectorize_hint = True  # 向量化提示

    def _validate_bounds(self) -> None:
        """验证向量在源数据边界内"""
        if len(self.offset) != len(self.origin.shape):
            raise ValueError(
                f"偏移维度 {len(self.offset)} 与源数据维度 {len(self.origin.shape)} 不匹配"
            )

        # 检查指定轴上的范围
        if (
            self.offset[self.axis] + self.shape[self.axis]
            > self.origin.shape[self.axis]
        ):
            raise ValueError(
                f"轴 {self.axis} 上的向量 ({self.offset[self.axis]}:{self.offset[self.axis]+self.shape[self.axis]}) "
                f"超出源数据范围 (0:{self.origin.shape[self.axis]})"
            )

    def get_absolute_position(self, relative_idx: int) -> Tuple[int, ...]:
        """计算相对于源数据的绝对位置"""
        if relative_idx < 0 or relative_idx >= self.shape[self.axis]:
            raise IndexError(
                f"索引 {relative_idx} 超出向量范围 (0:{self.shape[self.axis]})"
            )

        absolute_indices = list(self.offset)
        absolute_indices[self.axis] += relative_idx
        return tuple(absolute_indices)

    def get_element(self, idx: int) -> Any:
        """获取特定元素（需要源数据有值）"""
        if not self.origin.has_value():
            raise ValueError("源数据没有值，无法获取元素")

        absolute_indices = self.get_absolute_position(idx)

        if isinstance(self.origin, Tile):
            # 如果源是Tile，需要进一步转换到Tensor坐标
            return self.origin.get_element(*absolute_indices)
        else:
            # 如果源是Tensor
            origin_value = self.origin.get_value()
            return origin_value[absolute_indices]

    def set_unroll_hint(self, hint: bool = True) -> None:
        """设置循环展开提示"""
        self.unroll_hint = hint

    def set_vectorize_hint(self, hint: bool = True) -> None:
        """设置向量化提示"""
        self.vectorize_hint = hint

    def get_elements(self) -> List[Any]:
        """获取向量中的所有元素"""
        return [self.get_element(i) for i in range(self.shape[self.axis])]


# 辅助函数
def create_tensor_from_numpy(array: np.ndarray, name: Optional[str] = None) -> Tensor:
    """从NumPy数组创建张量"""
    return Tensor(
        shape=array.shape,
        dtype=array.dtype,
        name=name,
        layout=DataLayout.NCHW,  # 默认NCHW，可能需要根据实际情况调整
        memory_level=MemoryLevel.DRAM,
    )


def create_tile_from_tensor(
    tensor: Tensor,
    shape: Tuple[int, ...],
    offsets: Tuple[int, ...],
    name: Optional[str] = None,
    memory_level: MemoryLevel = MemoryLevel.L2_CACHE,
) -> Tile:
    """从张量创建数据块"""
    return Tile(
        shape=shape,
        dtype=tensor.dtype,
        origin_tensor=tensor,
        offsets=offsets,
        name=name,
        memory_level=memory_level,
    )


def create_vector_from_tile(
    tile: Tile,
    length: int,
    offset: Tuple[int, ...],
    axis: int = 0,
    name: Optional[str] = None,
    memory_level: MemoryLevel = MemoryLevel.REGISTER,
) -> Vector:
    """从数据块创建向量"""
    return Vector(
        length=length,
        dtype=tile.dtype,
        origin_tile=tile,
        offset=offset,
        axis=axis,
        name=name,
        memory_level=memory_level,
    )
