from Aipiler.dim import Dim
from Aipiler.datatype import DataType
import Aipiler.datatype as dtypes
from typing import Optional, List, Dict, Sequence, Tuple, Union
import numpy as np
from Aipiler.runtime import Storage
from Aipiler.runtime.device import Device, to_device
import torch


class FakeTensor:

    def __init__(
        self,
        symbolic_shape: Sequence[Dim],
        dtype: DataType,
        trace=None,
    ):
        from Aipiler.primitive import EinsumPrimitive

        self.symbolic_shape = symbolic_shape
        self.dtype = dtype
        for idx, dim in enumerate(self.symbolic_shape):
            dim.set_fake_tensor(self, idx)
        self._trace: Optional[EinsumPrimitive] = trace


# TODO: data layout
class Tensor:
    def __init__(
        self,
        dtype: DataType,
        device: Union[Device, str],
        storage: Storage,
        shape: Optional[Sequence[int]] = None,
    ) -> None:

        self.dtype = dtype
        self.device = to_device(device) if isinstance(device, str) else device
        self.storage = storage
        self._shape = shape

    @property
    def dim(self):
        return len(self._shape)

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
            s.insert(0, stride)
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


def from_torch(torch_tensor: torch.Tensor) -> Tensor:

    if not isinstance(torch_tensor, torch.Tensor):
        raise ValueError("Expect a torch.Tensor, got {}".format(type(torch_tensor)))
    if torch_tensor.requires_grad:
        raise ValueError("Unsupport torch tensor that requires grad")
    assert isinstance(torch_tensor, torch.Tensor)
    if torch_tensor.dtype == torch.bool:
        return from_dlpack(torch_tensor.to(dtype=torch.uint8)).to(dtype="bool")
    else:
        return from_dlpack(torch_tensor)


def from_torch_to_fake_tensor(torch_tensor: torch.Tensor) -> FakeTensor:
    from Aipiler.utils.dlpack import DLDataType

    return FakeTensor(
        symbolic_shape=[Dim() for _ in range(torch_tensor.dim())],
        dtype=dtypes.f32,
        trace=None,
    )


def empty(
    shape: Sequence[int],
    dtype: Union[DataType, str] = "float32",
    device: Union[Device, str] = "cpu",
):
    dtype = dtypes.data_type(dtype)
    shape = list(shape)
    # malloc memory
    size = 1
    for i in shape:
        size *= i
    num_bytes = int(size * dtype.nbytes)
    storage = Storage.new(device, num_bytes)

    symbolic_shape = [Dim() for _ in range(len(shape))]
    return Tensor(
        symbolic_shape=symbolic_shape,
        dtype=dtype,
        device=device,
        storage=storage,
        shape=shape,
    )
