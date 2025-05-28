import ctypes
from Aipiler import datatype as dtypes

_dtypes_mapping = {
    dtypes.int8: ctypes.c_int8,
    dtypes.int16: ctypes.c_int16,
    dtypes.int32: ctypes.c_int32,
    dtypes.int64: ctypes.c_int64,
    dtypes.uint8: ctypes.c_uint8,
    dtypes.uint16: ctypes.c_uint16,
    dtypes.uint32: ctypes.c_uint32,
    dtypes.uint64: ctypes.c_uint64,
    # dtypes.float16: no float16 in ctypes for now, we might need a custom type
    dtypes.float32: ctypes.c_float,
    dtypes.float64: ctypes.c_double,
    dtypes.boolean: ctypes.c_bool,
}


def getMemrefType(dtype: dtypes.DataType, isDynamic: bool = False) -> ctypes.Structure:
    if dtype not in _dtypes_mapping:
        raise ValueError(f"Unsupport datatype: {dtype.name}")

    cty = _dtypes_mapping[dtype]
    if isDynamic:
        cls_name = "DynamicMemRefType_" + dtype.name
        _fields_ = [
            ("rank", ctypes.c_int64),
            ("basePtr", ctypes.POINTER(cty)),
            ("data", ctypes.POINTER(cty)),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
        ]
    else:
        cls_name = "RankedMemRefType_" + dtype.name
        _fields_ = [
            ("basePtr", ctypes.POINTER(cty)),
            ("data", ctypes.POINTER(cty)),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
        ]

    cls = type(
        cls_name,
        (ctypes.Structure,),
        {
            "_fields_": _fields_,
            "__module__": __name__,
        },
    )
    return cls
