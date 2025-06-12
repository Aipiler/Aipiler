import torch
from torch import nn
from torch.export import export
import Aipiler
from iree.turbine import aot
import iree.runtime as rt
import numpy as np


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class RMSNorm_Matmul(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.rmsnorm = Qwen2RMSNorm(hidden_size, eps)

    def forward(self, x, weight):
        x = self.rmsnorm(x)
        return torch.matmul(x, weight)


X = torch.randn(16, 1024, dtype=torch.float32)
W = torch.randn(1024, 4096, dtype=torch.float32)
example_args = (X, W)

model = RMSNorm_Matmul(hidden_size=1024).cpu()

# exported_program: torch.export.ExportedProgram = export(model, args=example_args)

exported = aot.export(model, args=example_args)
compiled_binary = exported.compile(save_to=None)

config = rt.Config("local-task")
vmm = rt.load_vm_module(
    rt.VmModule.copy_buffer(config.vm_instance, compiled_binary.map_memory()),
    config,
)

i = 16
hidden_size = 1024
j = 4096

a = np.random.rand(i, hidden_size).astype(np.float32)
w_rms = np.random.rand(hidden_size).astype(np.float32)
w_mm = np.random.rand(hidden_size, j).astype(np.float32)

y = vmm.main(a, w_mm).to_host()
torch_y = model(torch.from_numpy(a), torch.from_numpy(w_mm))
np_y = torch_y.detach().numpy()

print("最大绝对误差：\t", np.max(np.abs(np_y - y)))

safe_div = np.where(np.abs(np_y) > 0, np.abs(np_y), 1)
rel_error_safe = np.abs(np_y - y) / safe_div
print("最大相对误差：\t", np.max(rel_error_safe))
# TODO: 误差较大
# ​绝对误差​：不超过 1e-6（理想目标 1e-7）
# ​相对误差​：不超过 1e-5（严格目标 1e-6）
