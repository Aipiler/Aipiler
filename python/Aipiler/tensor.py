from Aipiler.dim import Dim
from Aipiler.datatype import DataType
import Aipiler.datatype as dtypes
from typing import Optional, List, Dict, Sequence, Tuple, Union
import numpy as np
from Aipiler.runtime_Aipiler import Storage
from Aipiler.runtime_Aipiler.device import Device, to_device
import torch


class FakeData:
    def __init__(self, dtype: DataType):
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype


class FakeScalar(FakeData):
    def __init__(self, sym_val: Union[Dim, int, float], dtype: DataType):
        super().__init__(dtype=dtype)
        self._sym_val = sym_val

    @property
    def sym_val(self):
        return self._sym_val


class FakeTensor(FakeData):
    def __init__(
        self,
        symbolic_shapes: Sequence[Dim],
        dtype: DataType,
        trace=None,
    ):
        from Aipiler.primitive import EinsumPrimitive

        super().__init__(dtype=dtype)

        self.symbolic_shapes = symbolic_shapes
        for idx, dim in enumerate(self.symbolic_shapes):
            dim._fake_tensor = self
            dim._idx_in_tensor = idx
        self._trace: Optional[EinsumPrimitive] = trace

    def get_dim(self, idx: int) -> Dim:
        return self.symbolic_shapes[idx]

    def replace_dim(self, idx: int, dim: Dim):
        dim._fake_tensor = self
        dim._idx_in_tensor = idx
        self.symbolic_shapes[idx] = dim

    def dim(self):
        """
        Returns the number of dimensions of self tensor.
        """
        return len(self.symbolic_shapes)


# TODO: data layout
class Tensor:
    def __init__(
        self,
        shape: Sequence[int],
        dtype: Union[DataType, str],
        device: Union[Device, str],
        storage: Storage,
    ) -> None:
        self._shape = list(shape)
        self.dtype = dtypes.data_type(dtype) if isinstance(device, str) else dtype
        self.device = to_device(device) if isinstance(device, str) else device
        self.storage = storage

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

    return FakeTensor(
        symbolic_shapes=[Dim(torch_tensor.shape[i]) for i in range(torch_tensor.dim())],
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

    return Tensor(
        shape=shape,
        dtype=dtype,
        device=device,
        storage=storage,
    )
