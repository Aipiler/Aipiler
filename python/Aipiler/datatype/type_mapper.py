import torch
from typing import Dict
from .dataType import DataType
import Aipiler
import mlir.ir as ir


# TODO: replace with importing tensor from dlpack
class DtypeMapper:
    """PyTorch dtype 与自定义 Dtype 的映射器"""

    # 类级别的映射表，只初始化一次
    _PYTORCH_TO_AIPILER: Dict[torch.dtype, DataType] = {
        torch.float32: Aipiler.f32,
        torch.float: Aipiler.f32,
        torch.float64: Aipiler.f64,
        torch.double: Aipiler.f64,
        torch.int32: Aipiler.i32,
        torch.int: Aipiler.i32,
        torch.int64: Aipiler.i64,
        torch.long: Aipiler.i64,
        torch.bool: Aipiler.boolean,
        torch.uint8: Aipiler.uint8,
        torch.int8: Aipiler.i8,
        torch.int16: Aipiler.i16,
        torch.short: Aipiler.i16,
    }

    # 反向映射表（如果需要的话）
    _AIPILER_TO_PYTORCH: Dict[DataType, torch.dtype] = {
        Aipiler.f32: torch.float32,
        Aipiler.f64: torch.float64,
        Aipiler.i32: torch.int32,
        Aipiler.i64: torch.int64,
        Aipiler.boolean: torch.bool,
        Aipiler.uint8: torch.uint8,
        Aipiler.i8: torch.int8,
        Aipiler.i16: torch.int16,
        # UINT16, UINT32, UINT64 在 PyTorch 中不存在
    }

    _AIPILER_TO_MLIR: Dict[DataType, str] = {
        Aipiler.f32: ir.F32Type.get(),
        Aipiler.boolean: ir.IntegerType.get_signless(1),
    }

    @classmethod
    def from_pytorch(cls, pytorch_dtype: torch.dtype) -> DataType:
        """从 PyTorch dtype 转换到aipiler Dtype"""
        if pytorch_dtype not in cls._PYTORCH_TO_AIPILER:
            raise ValueError(f"不支持的 PyTorch dtype: {pytorch_dtype}")
        return cls._PYTORCH_TO_AIPILER[pytorch_dtype]

    @classmethod
    def to_pytorch(cls, aipiler_dtype: DataType) -> torch.dtype:
        """从aipiler Dtype 转换到 PyTorch dtype"""
        if aipiler_dtype not in cls._AIPILER_TO_PYTORCH:
            raise ValueError(f"无法转换到 PyTorch dtype: {aipiler_dtype}")
        return cls._AIPILER_TO_PYTORCH[aipiler_dtype]

    @classmethod
    def to_mlir(cls, aipiler_dtype: DataType) -> torch.dtype:
        """从aipiler Dtype 转换到 MLIR dtype"""
        if aipiler_dtype not in cls._AIPILER_TO_MLIR:
            raise ValueError(f"无法转换到 MLIR dtype: {aipiler_dtype}")
        return cls._AIPILER_TO_MLIR[aipiler_dtype]

    @classmethod
    def is_supported_pytorch_dtype(cls, pytorch_dtype: torch.dtype) -> bool:
        """检查是否支持该 PyTorch dtype"""
        return pytorch_dtype in cls._PYTORCH_TO_AIPILER

    @classmethod
    def is_supported_aipiler_dtype(cls, custom_dtype: DataType) -> bool:
        """检查是否支持该自定义 Dtype"""
        return custom_dtype in cls._AIPILER_TO_PYTORCH

    @classmethod
    def get_supported_pytorch_dtypes(cls) -> list[torch.dtype]:
        """获取所有支持的 PyTorch dtype"""
        return list(cls._PYTORCH_TO_AIPILER.keys())

    @classmethod
    def get_supported_aipiler_dtypes(cls) -> list[DataType]:
        """获取所有支持的自定义 Dtype"""
        return list(cls._AIPILER_TO_PYTORCH.keys())
