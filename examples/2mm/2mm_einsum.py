from Aipiler.dsl import map, reduce, unary, einsum_env, einsum, compile_module
from Aipiler.tensor import FakeTensor, FakeScalar
from Aipiler.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
from Aipiler.dim import dim, dims
from Aipiler.datatype import DataType, f32
import Aipiler.datatype as dtypes
from Aipiler import aot
import iree.runtime as rt
import logging
import numpy as np
import unittest
import os
import statistics
import re


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@einsum
def mm(A: FakeTensor, B: FakeTensor):
    t = map(A, B, "ik, kj -> ikj", [], compute_op_str="*")
    return reduce(t, "ikj -> ij", ["k"], compute_op_str="+")


@einsum
def tmm(
    A: FakeTensor,
    w1: FakeTensor,
    w2: FakeTensor,
):
    t1 = map(A, w1, "ik, kj -> ikj", [], compute_op_str="*")
    t2 = map(t1, w2, "ikj, jl -> ikjl", [], compute_op_str="*")
    # t3 = reduce(t2, "ikjl -> il", ["k", "j"], compute_op_str="+")
    return t2


@einsum
def tmm_norm(
    A: FakeTensor,
    w1: FakeTensor,
    w2: FakeTensor,
):
    t1 = mm(A, w1)
    t2 = mm(t1, w2)
    return t2


def run_inference(device: str = "host"):

    if device == "host":
        target_backend = "host"
        device = "local-task"
    elif device == "cuda":
        target_backend = "cuda"
    elif device == "rvv":
        target_backend = "rvv"
        device = "local-task"
    i = 16  # 定义全局变量以供静态编译使用
    hidden_size = 1024
    j = 4096
    A = FakeTensor(dims(i, hidden_size), f32)
    w1 = FakeTensor(dims(hidden_size, j), f32)
    w2 = FakeTensor(dims(j, hidden_size), f32)

    compiled_binary = compile_module(
        tmm,
        [A, w1, w2],
        target_backend=target_backend,  #  save=True
    )

    config = rt.Config(device)
    vm_module = rt.VmModule.copy_buffer(
        config.vm_instance, compiled_binary.map_memory()
    )

    np_a = np.random.rand(i, hidden_size).astype(np.float32)
    np_w_rms = np.random.rand(hidden_size, j).astype(np.float32)
    np_w_mm = np.random.rand(j, hidden_size).astype(np.float32)

    inputs = [np_a, np_w_rms, np_w_mm]

    vmm = rt.load_vm_module(
        vm_module,
        config,
    )

    y = vmm.main(*inputs)
    print("aipiler result: \n", y.to_host())

    print("*" * 99)

    def tmm_np(A: np.ndarray, w1: np.ndarray, w2: np.ndarray):
        t1 = np.einsum("ik,kj->ikj", A, w1)
        t2 = np.einsum("ikj,jl->ikjl", t1, w2)
        return t2
        # t1 = np.matmul(A, w1)
        # return np.matmul(t1, w2)

    np_y = tmm_np(*inputs)
    print("numpy result: \n", np_y)

    print("最大绝对误差：\t", np.max(np.abs(np_y - y.to_host())))

    safe_div = np.where(np.abs(np_y) > 0, np.abs(np_y), 1)
    rel_error_safe = np.abs(np_y - y.to_host()) / safe_div
    print("最大相对误差：\t", np.max(rel_error_safe))
    # TODO: 误差较大
    # ​绝对误差​：不超过 1e-6（理想目标 1e-7）
    # ​相对误差​：不超过 1e-5（严格目标 1e-6）


def do_bench(i: int, device: str = "host"):
    if device == "host":
        target_backend = "host"
        device = "local-task"
    elif device == "cuda":
        target_backend = "cuda"
    elif device == "rvv":
        target_backend = "rvv"
        device = "local-task"
    # i = 1024  # 定义全局变量以供静态编译使用
    hidden_size = 1024
    j = 4096
    A = FakeTensor(dims(i, hidden_size), f32)
    w1 = FakeTensor(dims(hidden_size, j), f32)
    w2 = FakeTensor(dims(j, hidden_size), f32)

    # A = FakeTensor(dims("i", "hidden_size"), f32)
    # rms_weight = FakeTensor(dims("hidden_size"), f32)
    # mm_weight = FakeTensor(dims("hidden_size", "j"), f32)

    compiled_binary = compile_module(
        tmm,
        [A, w1, w2],
        target_backend=target_backend,  #  save=True
    )

    config = rt.Config(device)
    vm_module = rt.VmModule.copy_buffer(
        config.vm_instance, compiled_binary.map_memory()
    )

    np_a = np.random.rand(i, hidden_size).astype(np.float32)
    np_w_rms = np.random.rand(hidden_size, j).astype(np.float32)
    np_w_mm = np.random.rand(j, hidden_size).astype(np.float32)

    inputs = [np_a, np_w_rms, np_w_mm]
    # run_inference(vm_module, config, inputs)
    benchmark_config = BenchmarkConfig(num_runs=20)
    benchmarker = BenchmarkRunner(benchmark_config)
    result = benchmarker.run_benchmark(
        vm_module,
        "main",
        inputs,
        f"{tmm.__name__}_{i}",
        device=device,
    )
    benchmarker.print_result_simple(result)


if __name__ == "__main__":

    run_inference(device="host")

    # i_list = (16,)
    # # i_list = (16, 32, 64, 128, 256, 512, 1024)
    # for i in i_list:
    #     print(f"i = {i}")
    #     do_bench(i, device="cuda")
