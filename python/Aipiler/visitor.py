from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
from Aipiler.primitive import (
    EinsumPrimitive,
    MapPrimitive,
    ReducePrimitive,
    PopulatePrimitive,
    UnaryPrimitive,
)
from Aipiler.tensor import Tensor
from Aipiler.datatype import DtypeMapper
from mlir import ir
from mlir.dialects.linalg.opdsl.lang import *
from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import func
from mlir.dialects import linalg
from mlir.dialects import tensor


class Visitor(ABC):

    @abstractmethod
    def visit_map(self, node: MapPrimitive) -> Any:
        pass

    @abstractmethod
    def visit_reduce(self, node: ReducePrimitive) -> Any:
        pass

    @abstractmethod
    def visit_populate(self, node: PopulatePrimitive) -> Any:
        pass

    @abstractmethod
    def visit_unary(self, node: UnaryPrimitive) -> Any:
        pass


class MLIRCodeGenVisitor(Visitor):

    def __init__(
        self, context: ir.Context, symbol_table: Dict[Tensor, ir.Value]
    ) -> None:
        self.visited_nodes: List[EinsumPrimitive] = []
        self.symbol_table: Dict[Tensor, ir.Value] = symbol_table
        self.context: ir.Context = context

    def visit_map(
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
        for script in node.iteration_scripts:
            symbol_defs[script] = getattr(S, script)
            domain_defs[script] = getattr(D, script)

        @linalg_structured_op
        def _map(
            A=TensorDef(T, *(symbol_defs[s] for s in node.lhs_scripts)),
            B=TensorDef(T, *(symbol_defs[s] for s in node.rhs_scripts)),
            C=TensorDef(
                T,
                *(symbol_defs[s] for s in node.output_scripts),
                output=True,
            ),
        ):
            domain((domain_defs[s] for s in node.iteration_scripts))
            output_indices = tuple(domain_defs[s] for s in node.output_scripts)
            lhs_indices = tuple(domain_defs[s] for s in node.lhs_scripts)
            rhs_indices = tuple(domain_defs[s] for s in node.rhs_scripts)
            # TODO: 当前只支持加减乘数,不能写死
            C[output_indices] = A[lhs_indices] * B[rhs_indices]

        mlir_dtype = DtypeMapper.to_mlir(node.output.dtype)
        # TODO: 需要处理symbolic shape
        init_result = tensor.EmptyOp(node.output.shape, mlir_dtype)
        op = _map(
            first_value,
            second_value,
            outs=[init_result.result],
        )

        return op

    @linalg_structured_op
    def visit_reduce(self, node: ReducePrimitive) -> ir.Value:
        self.visited_nodes.append(node)

        # 从符号表中找到输入张量的value
        input_tensors = node.inputs
        input_value = self.symbol_table[input_tensors[0]]
        if input_value not in self.symbol_table:
            raise ValueError(f"Input tensor {input_value} not found in symbol table.")

        # 根据einsum_str 构建linalg.generic op
        symbol_defs = {}
        domain_defs = {}
        for script in node.iteration_scripts:
            symbol_defs[script] = getattr(S, script)
            domain_defs[script] = getattr(D, script)

        @linalg_structured_op
        def _map(
            INPUT=TensorDef(T, *(symbol_defs[s] for s in node.x_scripts)),
            OUTPUT=TensorDef(
                T,
                *(symbol_defs[s] for s in node.output_scripts),
                output=True,
            ),
        ):
            domain(*(domain_defs[s] for s in node.iteration_scripts))
            output_indices = tuple(domain_defs[s] for s in node.output_scripts)
            input_indices = tuple(domain_defs[s] for s in node.x_scripts)
            # TODO: 当前只支持加减乘数,不能写死
            OUTPUT[output_indices] += INPUT[input_indices]

        mlir_dtype = DtypeMapper.to_mlir(node.output.dtype)
        # TODO: 需要处理symbolic shape
        init_result = tensor.EmptyOp(node.output.shape, mlir_dtype)
        op = _map(
            input_value,
            outs=[init_result.result],
        )

        return op

    def visit_populate(self, node: PopulatePrimitive) -> ir.Value:
        self.visited_nodes.append(node)
        return node.output

    def visit_unary(self, node: UnaryPrimitive) -> ir.Value:
        self.visited_nodes.append(node)
        return node.output
