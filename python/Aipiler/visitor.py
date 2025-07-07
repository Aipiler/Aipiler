from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
from Aipiler.primitive import (
    EinsumPrimitive,
    MapPrimitive,
    ReducePrimitive,
    PopulatePrimitive,
    UnaryPrimitive,
    CascadePrimitive,
)
from Aipiler.tensor import Tensor
from Aipiler import datatype as dtypes
from iree.compiler import ir
from iree.compiler.dialects.linalg.opdsl.lang import *
from iree.compiler.ir import *
from iree.compiler.dialects import builtin
from iree.compiler.dialects import func
from iree.compiler.dialects import linalg
from iree.compiler.dialects import tensor


class MLIRCodeGenVisitor:

    def __init__(
        self, context: ir.Context, symbol_table: Dict[Tensor, ir.Value]
    ) -> None:
        self.visited_nodes: List[EinsumPrimitive] = []
        self.symbol_table: Dict[Tensor, ir.Value] = symbol_table
        self.context: ir.Context = context
        with self.context:
            self._AIPILER_TO_MLIR: Dict[dtypes.DataType, str] = {
                dtypes.f32: ir.F32Type.get(),
                dtypes.boolean: ir.IntegerType.get_signless(1),
            }

            self._MLIR_TO_AIPILER: Dict[str, dtypes.DataType] = {
                ir.F32Type.get(): dtypes.f32,
                ir.IntegerType.get_signless(1): dtypes.boolean,
            }

    def from_dtype(self, dtype: dtypes.DataType):
        if dtype not in self._AIPILER_TO_MLIR:
            raise RuntimeError("Unsupported data type: {} now".format(dtype.name))
        return self._AIPILER_TO_MLIR[dtype]

    def to_dtype(self, mlirty: str) -> dtypes.DataType:
        if mlirty not in self._MLIR_TO_AIPILER:
            raise RuntimeError("Unsupported data type: {} now".format(mlirty))
        return self._MLIR_TO_AIPILER[mlirty]

    def visit_MapPrimitive(
        self,
        node: MapPrimitive,
    ) -> ir.Value:
        self.visited_nodes.append(node)

        # 从符号表中找到输入张量的value
        input_tensors = node.inputs
        first_value = self.symbol_table[input_tensors[0]]
        second_value = self.symbol_table[input_tensors[1]]
        if (
            first_value not in self.symbol_table
            or second_value not in self.symbol_table
        ):
            raise ValueError(
                f"Input tensor {first_value} and {second_value} not found in symbol table."
            )

        # 根据einsum_str 构建linalg.generic op
        symbol_defs = {}
        domain_defs = {}
        for script in node.iteration_axes:
            symbol_defs[script] = getattr(S, script)
            domain_defs[script] = getattr(D, script)

        @linalg_structured_op
        def _map(
            A=TensorDef(T, *(symbol_defs[s] for s in node.lhs_axes)),
            B=TensorDef(T, *(symbol_defs[s] for s in node.rhs_axes)),
            C=TensorDef(
                T,
                *(symbol_defs[s] for s in node.output_axes),
                output=True,
            ),
        ):
            domain((domain_defs[s] for s in node.iteration_axes))
            output_indices = tuple(domain_defs[s] for s in node.output_axes)
            lhs_indices = tuple(domain_defs[s] for s in node.lhs_axes)
            rhs_indices = tuple(domain_defs[s] for s in node.rhs_axes)
            # TODO: 当前只支持加减乘数,不能写死
            C[output_indices] = A[lhs_indices] * B[rhs_indices]

        mlir_dtype = self.from_dtype(node.output.dtype)
        # TODO: 需要处理symbolic shape
        init_result = tensor.EmptyOp(node.output.shape, mlir_dtype)
        op = _map(
            first_value,
            second_value,
            outs=[init_result.result],
        )

        return op

    def visit_ReducePrimitive(self, node: ReducePrimitive) -> ir.Value:
        self.visited_nodes.append(node)

        # 从符号表中找到输入张量的value
        input_tensors = node.inputs
        input_value = self.symbol_table[input_tensors[0]]
        if input_value not in self.symbol_table:
            raise ValueError(f"Input tensor {input_value} not found in symbol table.")

        # 根据einsum_str 构建linalg.generic op
        symbol_defs = {}
        domain_defs = {}
        for script in node.iteration_axes:
            symbol_defs[script] = getattr(S, script)
            domain_defs[script] = getattr(D, script)

        @linalg_structured_op
        def _map(
            INPUT=TensorDef(T, *(symbol_defs[s] for s in node.x_axes)),
            OUTPUT=TensorDef(
                T,
                *(symbol_defs[s] for s in node.output_axes),
                output=True,
            ),
        ):
            domain(*(domain_defs[s] for s in node.iteration_axes))
            output_indices = tuple(domain_defs[s] for s in node.output_axes)
            input_indices = tuple(domain_defs[s] for s in node.x_axes)
            # TODO: 当前只支持加减乘数,不能写死
            OUTPUT[output_indices] += INPUT[input_indices]

        mlir_dtype = self.from_dtype(node.output.dtype)
        # TODO: 需要处理symbolic shape
        init_result = tensor.EmptyOp(node.output.shape, mlir_dtype)
        op = _map(
            input_value,
            outs=[init_result.result],
        )

        return op

    def visit_PopulatePrimitive(self, node: PopulatePrimitive) -> ir.Value:
        self.visited_nodes.append(node)
        return node.output

    def visit_UnaryPrimitive(self, node: UnaryPrimitive) -> ir.Value:
        self.visited_nodes.append(node)
        return node.output

    def visit_CascadePrimitive(self, node: CascadePrimitive) -> ir.Value:
        self.visited_nodes.append(node)

        return node.output
