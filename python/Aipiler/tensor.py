from Aipiler.dim import Dim
from Aipiler.datatype import DataType
import Aipiler.datatype as dtypes
from typing import Optional, List, Dict, Sequence, Tuple
import torch
import numpy as np
from enum import Enum
import Aipiler
from Aipiler.runtime import Storage
from Aipiler.runtime.device import Device


class Tensor:
    def __init__(
        self,
        symbolic_shape: Sequence[Dim],
        dtype: DataType,
        device: Device,
        storage: Storage,
        shape: Optional[Sequence[int]] = None,
        trace=None,
    ) -> None:
        from Aipiler.primitive import EinsumPrimitive

        self._symbolic_shape = list(symbolic_shape)
        self.dtype = dtype
        self.device = device
        self.storage = storage
        self._shape = shape
        self._trace: Optional[EinsumPrimitive] = trace

    @property
    def dim(self):
        return len(self._symbolic_shape)

    @property
    def symbolic_shape(self):
        return self._symbolic_shape

    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        s = []
        if self._shape is None:
            raise RuntimeError()
        stride = 1
        for i in self.shape:
            s.insert(0, i)
            stride *= i
        return s

    def numpy(self) -> np.ndarray:
        if self.device.kind != "cpu":
            raise RuntimeError(
                "Cannot convert a tensor on {} to numpy array.".format(self.device)
            )
        if self.dtype in [dtypes.bfloat16, dtypes.tfloat32]:
            raise RuntimeError(
                "numpy does not support {}, converting to float32".format(
                    self.dtype.name
                )
            )
        if self.dtype == dtypes.boolean:
            # workaround for numpy not supporting exporting boolean to dlpack
            return np.from_dlpack(self.to(dtype="uint8")).astype(np.bool_)
        else:
            return np.from_dlpack(self)

    def __str__(self):
        head = "Tensor(shape={}, dtype='{}', device='{}')".format(
            self.shape, self.dtype.name, self.device
        )
        if self.storage:
            array_str = str(self.numpy())
            return "{}\n{}".format(head, array_str)
        else:
            if self._trace is None:
                return head
            else:
                return "{}\nfrom {}".format(head, self._trace)

    def __dlpack__(self, stream: Optional[int] = None):
        from .utils.dlpack import to_dlpack

        # TODO: 根据stream参数处理CUDA流的同步
        return to_dlpack(self)

    def __dlpack_device__(self) -> Tuple[int, int]:
        from .utils.dlpack import to_dlpack_device

        return to_dlpack_device(self)


def from_dlpack(dltensor) -> Tensor:
    """
    Create a aipiler tensor from an object that implements the __dlpack__ protocol.

    Parameters
    ----------
    dltensor: an object that implements the DLPack protocol.
        The object must have the method `__dlpack__` that returns a PyCapsule object with name `dltensor`.

    Returns
    -------
    ret: Tensor
        The aipiler tensor that shares the same storage with the DLPack tensor.
    """
    from .utils.dlpack import from_dlpack_capsule

    if not hasattr(dltensor, "__dlpack__"):
        raise RuntimeError("Expect a dltensor that implements __dlpack__ method.")

    return from_dlpack_capsule(dltensor.__dlpack__())


def from_torch(torch_tensor):
    import torch

    if not isinstance(torch_tensor, torch.Tensor):
        raise ValueError("Expect a torch.Tensor, got {}".format(type(torch_tensor)))
    if torch_tensor.requires_grad:
        raise ValueError("Unsupport torch tensor that requires grad")
    assert isinstance(torch_tensor, torch.Tensor)
    if torch_tensor.dtype == torch.bool:
        return from_dlpack(torch_tensor.to(dtype=torch.uint8)).to(dtype="bool")
    else:
        return from_dlpack(torch_tensor)
