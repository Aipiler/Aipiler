from typing import Union
from .dataType import DataType
from .integer import (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
)
from .float import (
    float16,
    float32,
    float64,
    bfloat16,
    tfloat32,
    f16,
    f32,
    f64,
    bf16,
    tf32,
)
from .bool import boolean

AIPILER_TYPES = (
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    float16,
    float32,
    float64,
    bfloat16,
    tfloat32,
    f16,
    f32,
    f64,
    bf16,
    tf32,
    boolean,
)

name2dtype = {dtype.name: dtype for dtype in AIPILER_TYPES}
sname2dtype = {dtype.short_name: dtype for dtype in AIPILER_TYPES}


def data_type(dtype: Union[str, DataType]) -> DataType:
    if isinstance(dtype, DataType):
        return dtype
    elif isinstance(dtype, str):
        if dtype in name2dtype:
            return name2dtype[dtype]
        elif dtype in sname2dtype:
            return sname2dtype[dtype]
        else:
            raise ValueError(
                "Unknown data type: {}, candidates:\n{}".format(
                    dtype, "\n".join(name2dtype.keys())
                )
            )
    else:
        raise ValueError(
            "Expect a string or a DataType, but got {}".format(type(dtype))
        )


def supported(name: str) -> bool:
    return name in name2dtype
