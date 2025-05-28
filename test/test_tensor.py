import Aipiler
from Aipiler import Tensor
from Aipiler import tensor
import torch


def test_from_torch():
    torch_a = torch.randn([2, 2], dtype=torch.float32)
    aipiler_a = tensor.from_torch(torch_a)
    assert len(aipiler_a.symbolic_shape) == 2
    assert aipiler_a.dtype is Aipiler.f32
    print(aipiler_a)


def test_empty():
    t = tensor.empty([2, 2])
    print(t)
