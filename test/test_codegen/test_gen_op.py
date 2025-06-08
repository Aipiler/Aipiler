# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import func
from mlir.dialects import linalg
from mlir.dialects import tensor

from mlir.dialects.linalg.opdsl.lang import *

T1 = TV.T1
T2 = TV.T2


@linalg_structured_op
def matmul_mono(
    A=TensorDef(T, S.M, S.K),
    B=TensorDef(T, S.K, S.N),
    C=TensorDef(T, S.M, S.N, output=True),
):
    domain(D.m, D.n, D.k)
    C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


iteration_scripts = ["I", "J", "K"]
A_scripts = ["I", "K"]
B_scripts = ["K", "J"]
C_scripts = ["I", "K", "J"]
D_scripts = ["I", "J"]
symbol_defs = {}
domain_defs = {}
for script in iteration_scripts:
    symbol_defs[script] = getattr(S, script)
    domain_defs[script] = getattr(D, script)


@linalg_structured_op
def _map(
    A=TensorDef(T, *(symbol_defs[s] for s in A_scripts)),
    B=TensorDef(T, *(symbol_defs[s] for s in B_scripts)),
    C=TensorDef(
        T,
        *(symbol_defs[s] for s in C_scripts),
        output=True,
    ),
):
    domain(*(domain_defs[s] for s in iteration_scripts))
    output_indices = tuple(domain_defs[s] for s in C_scripts)
    lhs_indices = tuple(domain_defs[s] for s in A_scripts)
    rhs_indices = tuple(domain_defs[s] for s in B_scripts)
    # TODO: 当前只支持加减乘数
    C[output_indices] = A[lhs_indices] * B[rhs_indices]


@linalg_structured_op
def _reduce(
    INPUT=TensorDef(T, *(symbol_defs[s] for s in C_scripts)),
    OUTPUT=TensorDef(
        T,
        *(symbol_defs[s] for s in D_scripts),
        output=True,
    ),
):
    domain(*(domain_defs[s] for s in iteration_scripts))
    output_indices = tuple(domain_defs[s] for s in D_scripts)
    input_indices = tuple(domain_defs[s] for s in C_scripts)
    # TODO: 当前只支持加减乘数
    OUTPUT[output_indices] += INPUT[input_indices]


def test_():

    with Context() as ctx, Location.unknown():
        module = Module.create()
        f16 = F16Type.get()
        f32 = F32Type.get()
        f64 = F64Type.get()
        i8 = IntegerType.get_signless(8)
        i16 = IntegerType.get_signless(16)
        i32 = IntegerType.get_signless(32)
        dyn = ShapedType.get_dynamic_size()
        with InsertionPoint(module.body):

            @func.FuncOp.from_py_func(
                RankedTensorType.get((4, dyn), f32), RankedTensorType.get((dyn, 8), f32)
            )
            def gen_map(A, B):
                init_result1 = tensor.EmptyOp([4, dyn, 8], f32)
                m1 = _map(A, B, outs=[init_result1.result])
                init_result2 = tensor.EmptyOp([4, 8], f32)
                r1 = _reduce(m1, outs=[init_result2.result])
                return r1

            # Multiplication indexing maps. We verify only the indexing maps of the
            # first multiplication and then do additional tests on casting and body
            # generation behavior.
            # CHECK: #[[$MUL_MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
            # CHECK: #[[$MUL_MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
            # CHECK: #[[$MUL_MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

            # CHECK-LABEL: func @test_matmul_mono
            # CHECK-SAME:  %[[A:.+]]: tensor<4x16xf32>
            # CHECK-SAME:  %[[B:.+]]: tensor<16x8xf32>
            # CHECK: %[[INITC:.+]] = tensor.empty() : tensor<4x8xf32>
            # CHECK: linalg.generic
            # CHECK-SAME: indexing_maps = [#[[$MUL_MAP_A]], #[[$MUL_MAP_B]], #[[$MUL_MAP_C]]]
            # CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
            # CHECK-SAME: ins(%[[A]], %[[B]]
            # CHECK-SAME: outs(%[[INITC]]
            # @func.FuncOp.from_py_func(
            #     RankedTensorType.get((4, 16), f32), RankedTensorType.get((16, 8), f32)
            # )
            # def test_matmul_mono(lhs, rhs):
            #     init_result = tensor.EmptyOp([4, 8], f32)
            #     return matmul_mono(lhs, rhs, outs=[init_result.result])

    print(module)
