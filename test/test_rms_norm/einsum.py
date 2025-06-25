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
def mean(A: FakeTensor):
    # A.mean(axis=-1, keepDim=True)
    t0 = reduce(A, "ij->i", "j", "+")
    dim = FakeScalar(
        A.get_dim(1),
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


i = 1024  # 定义全局变量以供静态编译使用
hidden_size = 1024
j = 4096
A = FakeTensor(dims(i, hidden_size), f32)
rms_weight = FakeTensor(dims(hidden_size), f32)
mm_weight = FakeTensor(dims(hidden_size, j), f32)
example_args = [A, rms_weight, mm_weight]

graph = einsum_env.compile(rms_norm_mm, example_args)
print(graph)
