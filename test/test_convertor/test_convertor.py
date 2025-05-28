import torch
import ctypes
from Aipiler.ffi import tensor_to_mlir_memref
import Aipiler
from Aipiler import datatype as dtypes
import os


def test_print():
    from Aipiler.tensor import from_torch

    _a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    a = from_torch(_a)
    print("aipiler tensor: ", a)
    print("aipiler storage = {}".format(hex(a.storage.addr)))

    print("memref: ")
    obj = tensor_to_mlir_memref(a)
    lib_path = os.path.dirname(os.path.abspath(__file__)) + "/libtest.so"
    _LIB_PRINT = ctypes.CDLL(lib_path)
    _LIB_PRINT.printMemrefI32(ctypes.byref(obj))


def test_matmul():
    from Aipiler.tensor import from_torch, empty

    _A = torch.randn([2, 2], dtype=torch.float32)
    _B = torch.randn([2, 2], dtype=torch.float32)

    A = from_torch(_A)
    B = from_torch(_B)
    C = empty([2, 2], dtype=dtypes.float32)

    A_obj = tensor_to_mlir_memref(A)
    B_obj = tensor_to_mlir_memref(B)
    C_obj = tensor_to_mlir_memref(C)

    lib_path = os.path.dirname(os.path.abspath(__file__)) + "/libtest.so"
    clib = ctypes.CDLL(lib_path)

    clib.matmul(ctypes.byref(A_obj), ctypes.byref(B_obj), ctypes.byref(C_obj))

    print("result from torch")
    print(torch.matmul(_A, _B))
    print()
    print("result from aipiler")
    print(C)


test_matmul()
