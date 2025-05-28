import torch
import ctypes
from Aipiler.ffi import tensor_to_mlir_memref
import Aipiler
import os


def print_tensor(tensor: Aipiler.Tensor):
    obj = tensor_to_mlir_memref(tensor)
    lib_path = os.path.dirname(os.path.abspath(__file__)) + "/libprint.so"
    _LIB_PRINT = ctypes.CDLL(lib_path)
    _LIB_PRINT.printMemref(ctypes.byref(obj))


def test_convertor():
    from Aipiler.tensor import from_torch

    _a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    a = from_torch(_a)
    print("aipiler tensor: ", a)
    print("aipiler storage = {}".format(hex(a.storage.addr)))

    print("memref: ")
    print_tensor(a)
