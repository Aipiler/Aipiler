from Aipiler.dsl import reduce, unary, map, einsum, compile_module
from Aipiler.tensor import FakeTensor, FakeScalar
from Aipiler.dim import dims
import Aipiler.datatype as dtypes
from typing import Optional
from Aipiler.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
import iree.runtime as rt
import numpy as np


@einsum
def gate_proj(x: FakeTensor, w: FakeTensor):
    t0 = map(x, w, "ijk, qk -> ijkq", "*")
    y = reduce(t0, "ijkq -> ijq", "k", "+")
    return y


@einsum
def up_proj(x: FakeTensor, w: FakeTensor):
    t0 = map(x, w, "ijk, qk -> ijkq", "*")
    y = reduce(t0, "ijkq -> ijq", "k", "+")
    return y


@einsum
def down_proj(x: FakeTensor, w: FakeTensor):
    t0 = map(x, w, "ijq, kq -> ijqk", "*")
    y = reduce(t0, "ijqk -> ijk", "q", "+")
    return y


@einsum
def silu(x: FakeTensor):
    # -x
    t0 = unary(x, "neg")
    # exp
    t1 = unary(t0, "exp")
    # 1.0 +
    c1 = FakeScalar(1, dtype=dtypes.float32)
    t2 = map(t1, c1, "ijk, _ -> ijk", "+")
    y = map(x, t2, "ijk, ijk -> ijk", "/")
    return y


@einsum
def DeepseekV3MLP_einsum(
    x: FakeTensor, w_gate: FakeTensor, w_up: FakeTensor, w_down: FakeTensor
):
    # self.gate_proj(x)
    t0 = gate_proj(x, w_gate)
    # self.act_fn(...)
    t1 = silu(t0)
    # self.up_proj(x)
    t2 = up_proj(x, w_up)
    # self.gate_proj(x) * self.up_proj(x)
    t3 = map(t1, t2, "ijq, ijq -> ijq", "*")
    y = down_proj(t3, w_down)
    return y


def DeepseekV3MLP_np(
    x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray
):
    t0 = np.matmul(x, w_gate.T)
    t1 = t0 / (1.0 + np.exp(-t0))  # silu
    t2 = np.matmul(x, w_up.T)
    t3 = t1 * t2
    y = np.matmul(t3, w_down.T)
    return y


def do_bench(device: str = "host"):
    if device == "host":
        target_backend = "host"
        device = "local-task"
    elif device == "cuda":
        target_backend = "cuda"
    elif device == "rvv":
        target_backend = "rvv"
        device = "local-task"
    batch_size = 1
    seq_len = 10
    hidden_size = 1024
    intermedia_size = 10
    x = FakeTensor(dims("batch_size", "seq_len", "hidden_size"), dtype=dtypes.float32)
    w_gate = FakeTensor(dims("intermedia_size", "hidden_size"), dtype=dtypes.float32)
    w_up = FakeTensor(dims("intermedia_size", "hidden_size"), dtype=dtypes.float32)
    w_down = FakeTensor(dims("hidden_size", "intermedia_size"), dtype=dtypes.float32)

    compiled_binary = compile_module(
        DeepseekV3MLP_einsum,
        # [x, w_gate, w_up],
        [x, w_gate, w_up, w_down],
        target_backend=target_backend,  #  save=True
    )

    config = rt.Config(device)
    vm_module = rt.VmModule.copy_buffer(
        config.vm_instance, compiled_binary.map_memory()
    )
    np_x = np.random.rand(batch_size, seq_len, hidden_size).astype(np.float32)
    np_w_gate = np.random.rand(intermedia_size, hidden_size).astype(np.float32)
    np_w_up = np.random.rand(intermedia_size, hidden_size).astype(np.float32)
    np_w_down = np.random.rand(hidden_size, intermedia_size).astype(np.float32)

    # inputs = [np_x, np_w_gate, np_w_up]
    inputs = [np_x, np_w_gate, np_w_up, np_w_down]
    run_inference(vm_module, config, inputs)
    return
    benchmark_config = BenchmarkConfig(num_runs=20)
    benchmarker = BenchmarkRunner(benchmark_config)
    result = benchmarker.run_benchmark(
        vm_module,
        "main",
        inputs,
        f"linear",
        device=device,
    )
    benchmarker.print_result_simple(result)


def run_inference(
    vm_module: rt.VmModule,
    config: rt.Config,
    inputs: list[np.ndarray],
):

    vmm = rt.load_vm_module(
        vm_module,
        config,
    )

    y = vmm.main(*inputs)
    print("aipiler result: \n", y.to_host())

    print("*" * 99)

    np_y = DeepseekV3MLP_np(*inputs)
    print("numpy result: \n", np_y)

    print("最大绝对误差：\t", np.max(np.abs(np_y - y.to_host())))

    safe_div = np.where(np.abs(np_y) > 0, np.abs(np_y), 1)
    rel_error_safe = np.abs(np_y - y.to_host()) / safe_div
    print("最大相对误差：\t", np.max(rel_error_safe))
    # ​绝对误差​：不超过 1e-6（理想目标 1e-7）
    # ​相对误差​：不超过 1e-5（严格目标 1e-6）


do_bench()
