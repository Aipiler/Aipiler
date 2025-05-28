import Aipiler
from Aipiler import Tensor
from Aipiler.ffi.memref import getMemrefType
import ctypes
import torch


def tensor_to_mlir_memref(tensor: Tensor):
    """
    Given a tensor obj, convert it to a MLIR DynamicMemref pointer can
    be used as a ctypes FFI argument.
    """
    assert tensor.shape is not None
    dtype = tensor.dtype
    shape = tensor.shape
    strides = tensor.strides
    memrefty = getMemrefType(tensor.dtype, isDynamic=True)
    obj = memrefty()
    cdtype = Aipiler.ffi.memref._dtypes_mapping[dtype]
    obj.rank = len(shape)
    obj.basePtr = ctypes.cast(tensor.storage.addr, ctypes.POINTER(cdtype))
    obj.data = ctypes.cast(tensor.storage.addr, ctypes.POINTER(cdtype))
    obj.offset = 0
    obj.sizes = (ctypes.c_int64 * len(shape))(*shape)
    obj.strides = (ctypes.c_int64 * len(strides))(*strides)
    return obj
