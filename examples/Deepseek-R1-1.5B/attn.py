from Aipiler.dsl import map, reduce, unary, einsum_env, einsum, compile_module
from Aipiler.tensor import FakeTensor, FakeScalar, Parameter, from_torch_to_parameter
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
from Aipiler import dsl
from Aipiler import datatype as dtypes
from typing import Optional

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Config:
    def __init__(self):
        self.hidden_size = 1536
        self.intermediate_size = 8960
        self.num_attention_heads = 12
        self.num_hidden_layers = 28
        self.hidden_act = "silu"
        self.max_position_embeddings = 131072
        self.rope_theta = 10000
        self.rms_norm_eps = 1e-6
        self.num_key_value_heads = 2

class Module:
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
        
    def forward(self, *args, **kwds):
        raise NotImplementedError()






def matmul(A: FakeTensor, B: FakeTensor):
    t = map(A, B, "ik, jk -> ijk", compute_op_str="*")
    return reduce(t, "ijk -> ij", ["k"], compute_op_str="+")


def rms_norm_mm(
    A: FakeTensor,
    rms_weight: FakeTensor,
    mm_weight: FakeTensor,
):
    t = qwen2_rms_norm(A, rms_weight)
    ret = mm(t, mm_weight)
    return ret





class SiLU(Module):
    def forward(self, x: FakeTensor):
        return unary(x, "ij -> ij", "silu")

class Qwen2MLP(Module):
    def __init__(self, config: Config):
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SiLU()

    def forward(self, x: FakeTensor):
        t1 = self.gate_proj(x)
        t2 = self.act_fn(t1)
        t3 = self.up_proj(x)
        t4 = map(t2, t3, "ij, ij -> ij", "*")
        t5 = self.down_proj(t4)
        return t5


class Qwen2RMSNorm(Module):
    def __init__(self, hidden_size, eps=1e-6):
        self.weight = Parameter(dims(hidden_size), dtypes.f32, storage=None)
        self.variance_epsilon = eps


    def forward(self, x: FakeTensor):
        
        
        def mean(A: FakeTensor):
            # A.mean(axis=-1, keepDim=True)
            t0 = reduce(A, "ij->i", "j", "+")
            dim = FakeScalar(
                A.get_dim(1),
                A.dtype,
            )
            t1 = map(t0, dim, "i, _ -> i", compute_op_str="/")
            return t1

        def qwen2_rms_norm(
            hidden_states: FakeTensor,
            weight: FakeTensor,
        ):
            assert hidden_states.dtype is dtypes.f32
            eps = FakeScalar(self.variance_epsilon, dtype=dtypes.f32)
            c2 = FakeScalar(2, dtype=dtypes.f32)
            tmp0 = map(hidden_states, c2, "ij, _ -> ij", "^")
            variance = mean(tmp0)
            # tmp1 = variance + eps
            tmp1 = map(variance, eps, "j, _ -> j", "+")
            # tmp2 = torch.rsqrt(tmp1)
            tmp2 = unary(tmp1, "rsqrt")
            hidden_states = map(hidden_states, tmp2, "ij, i -> ij", "*")
            # weight * hidden_state
            ret = map(weight, hidden_states, "j, ij -> ij", "*")
            return ret
        
        return qwen2_rms_norm(x, self.weight)


class Qwen2RotaryEmbedding(Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
    def forward(self, x: FakeTensor):
        # Implement the rotary embedding logic here
        pass  # Placeholder for actual implementation


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            dims(self.out_features, self.in_features),
            dtypes.f32, storage=None
        )
        if bias:
            self.bias = Parameter(
                dims(self.out_features),
                dtypes.f32,
                storage=None
            )
        else:
            self.bias = None

    def forward(self, x: FakeTensor):
        x = map(x, self.weight, "ik, jk -> ijk", compute_op_str="*")
        x = reduce(x, "ijk -> ij", ["k"], compute_op_str="+")
        if self.bias is not None:
            x = map(x, self.bias, "ij, j -> ij", "+")
        return x


class Softmax(Module):
    pass

class Qwen2Attention(Module):
    def __init__(self, config: Config, layer_idx: Optional[int] = None):
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
    def forward(self, hidden_states: FakeTensor):
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)