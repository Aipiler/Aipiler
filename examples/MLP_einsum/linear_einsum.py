from Aipiler.dsl import reduce, unary, map, einsum, compile_module
from Aipiler.tensor import FakeTensor, FakeScalar
from Aipiler.dim import dims
import Aipiler.datatype as dtypes
from typing import Optional
from Aipiler.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
import iree.runtime as rt
import numpy as np


@einsum
def linear(x: FakeTensor, w: FakeTensor, b: Optional[FakeTensor]):
    # t1 = x @ w
    t0 = map(x, w, "ijk, qk -> ijkq", "*")  # torch.linear need to transpose weight
    t1 = reduce(t0, "ijkq -> ijq", "k", "+")
    if b:
        # t2 = t1 + b
        t2 = map(t1, b, "ijq, q -> ijq", "+")
        return t2
    else:
        return t1


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
    intermedia_size = 1024
    x = FakeTensor(dims("batch_size", "seq_len", "hidden_size"), dtype=dtypes.float32)
    w = FakeTensor(dims("intermedia_size", "hidden_size"), dtype=dtypes.float32)
    b = FakeTensor(dims("intermedia_size"), dtype=dtypes.float32)

    compiled_binary = compile_module(
        linear,
        [x, w, b],
        target_backend=target_backend,  #  save=True
    )

    config = rt.Config(device)
    vm_module = rt.VmModule.copy_buffer(
        config.vm_instance, compiled_binary.map_memory()
    )

    np_x = np.random.rand(batch_size, seq_len, hidden_size).astype(np.float32)
    np_w = np.random.rand(intermedia_size, hidden_size).astype(np.float32)
    np_b = np.random.rand(intermedia_size).astype(np.float32)

    inputs = [np_x, np_w, np_b]
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

    def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray):
        t0 = np.matmul(x, w.T)
        if b is not None:
            return t0 + b
        else:
            return t0

    np_y = linear(*inputs)
    print("numpy result: \n", np_y)

    print("最大绝对误差：\t", np.max(np.abs(np_y - y.to_host())))

    safe_div = np.where(np.abs(np_y) > 0, np.abs(np_y), 1)
    rel_error_safe = np.abs(np_y - y.to_host()) / safe_div
    print("最大相对误差：\t", np.max(rel_error_safe))
    # ​绝对误差​：不超过 1e-6（理想目标 1e-7）
    # ​相对误差​：不超过 1e-5（严格目标 1e-6）


do_bench("cuda")
