import torch
from torch import Tensor
from typing import Optional
import torch
from torch import nn
from torch.export import export, Dim
from iree.turbine import aot
import iree.runtime as rt
import numpy as np
from Aipiler.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
import iree.runtime as rt
import numpy as np


class Linear(nn.Module):
    def __init__(self):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

    def forward(self, x, w, b):
        t0 = torch.matmul(x, w.T)
        if b is not None:
            return torch.add(t0, b)
        else:
            return t0


model = Linear()


def do_bench(device: str = "cuda"):
    if device == "host":
        target_backend = "llvm-cpu"
        device = "local-task"
    elif device == "cuda":
        target_backend = "cuda"
    elif device == "rvv":
        target_backend = "rvv"
        device = "local-task"
    # i = 1024
    hidden_size = 1024
    j = 4096

    x = torch.randn(i, hidden_size, dtype=torch.float32)
    rms_w = torch.randn(hidden_size, dtype=torch.float32)
    mm_w = torch.randn(hidden_size, j, dtype=torch.float32)
    example_args = (x, rms_w, mm_w)

    dim_i = Dim("i")
    dim_hidden_size = Dim("hidden_size")
    dim_j = Dim("j")
    dynamic_shapes = {
        "x": {0: dim_i, 1: dim_hidden_size},
        # 为 'rms_w' 指定动态维度
        "rms_w": {0: dim_hidden_size},
        # 为 'mm_w' 指定动态维度
        "mm_w": {0: dim_hidden_size, 1: dim_j},
    }

    # A = FakeTensor(dims("i", "hidden_size"), f32)
    # rms_weight = FakeTensor(dims("hidden_size"), f32)
    # mm_weight = FakeTensor(dims("hidden_size", "j"), f32)

    exported = aot.export(model, args=example_args, dynamic_shapes=dynamic_shapes)
    compiled_binary = exported.compile(save_to=None, target_backends=target_backend)

    config = rt.Config(device)
    vm_module = rt.VmModule.copy_buffer(
        config.vm_instance, compiled_binary.map_memory()
    )

    np_a = np.random.rand(i, hidden_size).astype(np.float32)
    np_w_rms = np.random.rand(hidden_size).astype(np.float32)
    np_w_mm = np.random.rand(hidden_size, j).astype(np.float32)

    inputs = [np_a, np_w_rms, np_w_mm]
    # run_inference(vm_module, config, inputs)
    benchmark_config = BenchmarkConfig(num_runs=20)
    benchmarker = BenchmarkRunner(benchmark_config)
    result = benchmarker.run_benchmark(
        vm_module,
        "main",
        inputs,
        f"{model.__class__.__name__}_{i}",
        device=device,
    )
    benchmarker.print_result_simple(result)


if __name__ == "__main__":
    i_list = (16, 32, 64, 128, 256, 512, 1024)
    # i_list = (16,)
    for i in i_list:
        print(f"i = {i}")
        do_bench(i, device="cuda")
