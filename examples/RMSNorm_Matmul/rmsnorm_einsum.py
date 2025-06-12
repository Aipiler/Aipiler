from Aipiler.dsl import map, reduce, unary, einsum_env, einsum
from Aipiler.tensor import FakeTensor, FakeScalar
from Aipiler.dim import dim, dims
from Aipiler.datatype import DataType, f32
import Aipiler.datatype as dtypes
from Aipiler import aot
import iree.runtime as rt
import logging
import numpy as np
import unittest

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@einsum
def mean(A: FakeTensor):
    # A.mean(axis=-1, keepDim=True)
    t0 = reduce(A, "ij->i", "j", "+")
    dim = FakeScalar(
        A.symbolic_shape[1],
        A.dtype,
    )
    t1 = map(t0, dim, "i, _ -> i", target_dim=["i"], compute_op_str="/")
    return t1


@einsum
def qwen2_rms_norm(
    hidden_states: FakeTensor,
    weight: FakeTensor,
):
    assert hidden_states.dtype is dtypes.f32
    eps = FakeScalar(1e-6, dtype=dtypes.f32)
    c2 = FakeScalar(2, dtype=dtypes.f32)
    tmp0 = map(hidden_states, c2, "ij, _ -> ij", [], "^")
    variance = mean(tmp0)
    # tmp1 = variance + eps
    tmp1 = map(variance, eps, "j, _ -> j", [], "+")
    # tmp2 = torch.rsqrt(tmp1)
    tmp2 = unary(tmp1, "rsqrt")
    hidden_states = map(hidden_states, tmp2, "ij, i -> ij", [], "*")
    # weight * hidden_state
    ret = map(weight, hidden_states, "j, ij -> ij", [], "*")
    return ret


@einsum
def mm(A: FakeTensor, B: FakeTensor):
    t = map(A, B, "ik, kj -> ikj", [], compute_op_str="*")
    return reduce(t, "ikj -> ij", ["k"], compute_op_str="+")


@einsum
def rms_norm_mm(
    A: FakeTensor,
    rms_weight: FakeTensor,
    mm_weight: FakeTensor,
):
    t = qwen2_rms_norm(A, rms_weight)
    ret = mm(t, mm_weight)
    return ret


def np_qwen2_rms_norm(A: np.ndarray, rms_weight: np.ndarray):
    eps = 1e-6
    tmp0 = np.power(A, 2)
    variance = np.mean(tmp0, axis=1, keepdims=True)
    tmp1 = np.add(variance, eps)
    tmp2 = np.divide(1.0, np.sqrt(tmp1))
    hidden_states = np.multiply(A, tmp2)
    rms_norm = np.multiply(rms_weight, hidden_states)
    return rms_norm


def np_rms_norm_mm(A: np.ndarray, rms_weight: np.ndarray, mm_weight: np.ndarray):
    rms_norm = np_qwen2_rms_norm(A, rms_weight)
    return np.matmul(rms_norm, mm_weight)


i = 1024
hidden_size = 1024
j = 4096
A = FakeTensor(dims("i", "hidden_size"), f32)
rms_weight = FakeTensor(dims("hidden_size"), f32)
mm_weight = FakeTensor(dims("hidden_size", "j"), f32)

a = np.random.rand(i, hidden_size).astype(np.float32)
w_rms = np.random.rand(hidden_size).astype(np.float32)
w_mm = np.random.rand(hidden_size, j).astype(np.float32)


graph = einsum_env.compile(rms_norm_mm, [A, rms_weight, mm_weight])
print("Graph: \n")
# print(graph)
print("\n")
exported = aot.export(graph)
print("MLIR: \n")
exported.print_readable()
print("\n")
compiled_binary = exported.compile(save_to=None)

config = rt.Config("local-task")
vmm = rt.load_vm_module(
    rt.VmModule.copy_buffer(config.vm_instance, compiled_binary.map_memory()),
    config,
)

y = vmm.main(a, w_rms, w_mm)
print("aipiler result: \n", y.to_host())

print("*" * 99)

np_y = np_rms_norm_mm(a, w_rms, w_mm)
print("numpy result: \n", np_y)

print("最大绝对误差：\t", np.max(np.abs(np_y - y.to_host())))

safe_div = np.where(np.abs(np_y) > 0, np.abs(np_y), 1)
rel_error_safe = np.abs(np_y - y.to_host()) / safe_div
print("最大相对误差：\t", np.max(rel_error_safe))
# TODO: 误差较大
# ​绝对误差​：不超过 1e-6（理想目标 1e-7）
# ​相对误差​：不超过 1e-5（严格目标 1e-6）
