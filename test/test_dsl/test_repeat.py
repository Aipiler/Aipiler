from Aipiler.dsl import repeat, einsum_env, einsum
from Aipiler.tensor import FakeTensor, FakeScalar
from Aipiler.dim import dim, dims
from Aipiler.datatype import DataType, f32
from Aipiler import aot
import iree.runtime as rt
import logging
import numpy as np
import unittest


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@einsum
def test_repeat(A: FakeTensor):
    return repeat(A, "b h w c -> b h n w c", n=2)


A = FakeTensor(dims("b", "h", "w", "c"), f32)
graph = einsum_env.compile(test_repeat, [A])
print("Graph: \n")
print(graph)
