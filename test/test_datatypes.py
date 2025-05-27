import Aipiler
from Aipiler import Tensor, from_torch_tensor
import torch


def test_from_torch():
    torch_a = torch.randn([2, 2], dtype=torch.float32)
    aipiler_a = from_torch_tensor(torch_a)
    assert len(aipiler_a.shape) == 2
    assert aipiler_a.dtype is Aipiler.f32
