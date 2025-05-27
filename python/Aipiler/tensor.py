from Aipiler.dim import Dim
from Aipiler.datatype import DataType
from typing import Optional, List, Dict
import torch
from enum import Enum


class Dtype(Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"
    UINT8 = "uint8"
    INT8 = "int8"
    INT16 = "int16"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"


class DtypeMapper:
    """PyTorch dtype 与自定义 Dtype 的映射器"""

    # 类级别的映射表，只初始化一次
    _PYTORCH_TO_CUSTOM: Dict[torch.dtype, Dtype] = {
        torch.float32: Dtype.FLOAT32,
        torch.float: Dtype.FLOAT32,
        torch.float64: Dtype.FLOAT64,
        torch.double: Dtype.FLOAT64,
        torch.int32: Dtype.INT32,
        torch.int: Dtype.INT32,
        torch.int64: Dtype.INT64,
        torch.long: Dtype.INT64,
        torch.bool: Dtype.BOOL,
        torch.uint8: Dtype.UINT8,
        torch.int8: Dtype.INT8,
        torch.int16: Dtype.INT16,
        torch.short: Dtype.INT16,
    }

    # 反向映射表（如果需要的话）
    _CUSTOM_TO_PYTORCH: Dict[Dtype, torch.dtype] = {
        Dtype.FLOAT32: torch.float32,
        Dtype.FLOAT64: torch.float64,
        Dtype.INT32: torch.int32,
        Dtype.INT64: torch.int64,
        Dtype.BOOL: torch.bool,
        Dtype.UINT8: torch.uint8,
        Dtype.INT8: torch.int8,
        Dtype.INT16: torch.int16,
        # UINT16, UINT32, UINT64 在 PyTorch 中不存在
    }

    @classmethod
    def from_pytorch(cls, pytorch_dtype: torch.dtype) -> Dtype:
        """从 PyTorch dtype 转换到自定义 Dtype"""
        if pytorch_dtype not in cls._PYTORCH_TO_CUSTOM:
            raise ValueError(f"不支持的 PyTorch dtype: {pytorch_dtype}")
        return cls._PYTORCH_TO_CUSTOM[pytorch_dtype]

    @classmethod
    def to_pytorch(cls, custom_dtype: Dtype) -> torch.dtype:
        """从自定义 Dtype 转换到 PyTorch dtype"""
        if custom_dtype not in cls._CUSTOM_TO_PYTORCH:
            raise ValueError(f"无法转换到 PyTorch dtype: {custom_dtype}")
        return cls._CUSTOM_TO_PYTORCH[custom_dtype]

    @classmethod
    def is_supported_pytorch_dtype(cls, pytorch_dtype: torch.dtype) -> bool:
        """检查是否支持该 PyTorch dtype"""
        return pytorch_dtype in cls._PYTORCH_TO_CUSTOM

    @classmethod
    def is_supported_custom_dtype(cls, custom_dtype: Dtype) -> bool:
        """检查是否支持该自定义 Dtype"""
        return custom_dtype in cls._CUSTOM_TO_PYTORCH

    @classmethod
    def get_supported_pytorch_dtypes(cls) -> list[torch.dtype]:
        """获取所有支持的 PyTorch dtype"""
        return list(cls._PYTORCH_TO_CUSTOM.keys())

    @classmethod
    def get_supported_custom_dtypes(cls) -> list[Dtype]:
        """获取所有支持的自定义 Dtype"""
        return list(cls._CUSTOM_TO_PYTORCH.keys())


class Tensor:
    def __init__(self, symbolic_shape: List[Dim], dtype: DataType, trace=None) -> None:
        from Aipiler.primitive import EinsumPrimitive

        self.symbolic_shape = symbolic_shape
        self.dtype = dtype
        self._trace: Optional[EinsumPrimitive] = trace

    @property
    def dim(self):
        return len(self.symbolic_shape)

    @property
    def shape(self):
        return self.symbolic_shape


def from_torch_tensor(tensor: torch.Tensor):
    # TODO: datatype of tensor
    assert isinstance(tensor, torch.Tensor)
    dim = tensor.dim()
    return Tensor([Dim() for _ in range(dim)], None)
