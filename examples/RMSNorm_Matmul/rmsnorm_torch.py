import torch
from torch import nn
from torch.export import export
from iree.turbine import aot
import iree.runtime as rt
import numpy as np
from Aipiler.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
import iree.runtime as rt
import numpy as np


# class Qwen2RMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         Qwen2RMSNorm is equivalent to T5LayerNorm
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.rand(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)

#     def extra_repr(self):
#         return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# class RMSNorm_Matmul(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         super().__init__()
#         self.rmsnorm = Qwen2RMSNorm(hidden_size, eps)

#     def forward(self, x, weight):
#         x = self.rmsnorm(x)
#         return torch.matmul(x, weight)


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, hidden_states, weight):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
        return weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class RMSNorm_Matmul(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.rmsnorm = Qwen2RMSNorm(hidden_size, eps)

    def forward(self, x, rms_w, mm_w):
        x = self.rmsnorm(x, rms_w)
        return torch.matmul(x, mm_w)


model = RMSNorm_Matmul(hidden_size=1024)


def do_bench(i: int):
    # i = 1024
    hidden_size = 1024
    j = 4096

    x = torch.randn(i, hidden_size, dtype=torch.float32)
    rms_w = torch.randn(hidden_size, dtype=torch.float32)
    mm_w = torch.randn(hidden_size, j, dtype=torch.float32)
    example_args = (x, rms_w, mm_w)

    # A = FakeTensor(dims("i", "hidden_size"), f32)
    # rms_weight = FakeTensor(dims("hidden_size"), f32)
    # mm_weight = FakeTensor(dims("hidden_size", "j"), f32)

    exported = aot.export(model, args=example_args)
    compiled_binary = exported.compile(save_to=None)

    config = rt.Config("local-task")
    vm_module = rt.VmModule.copy_buffer(
        config.vm_instance, compiled_binary.map_memory()
    )

    np_a = np.random.rand(i, hidden_size).astype(np.float32)
    np_w_rms = np.random.rand(hidden_size).astype(np.float32)
    np_w_mm = np.random.rand(hidden_size, j).astype(np.float32)

    inputs = [np_a, np_w_rms, np_w_mm]
    # run_inference(vm_module, config, inputs)
    config = BenchmarkConfig(num_runs=20)
    benchmarker = BenchmarkRunner(config)
    result = benchmarker.run_benchmark(
        vm_module, "main", inputs, f"{model.__class__.__name__}_{i}"
    )
    benchmarker.print_result_simple(result)


if __name__ == "__main__":
    i_list = (16, 32, 64, 128, 256, 512, 1024)
    # i_list = (16,)
    for i in i_list:
        print(f"i = {i}")
        do_bench(i)
