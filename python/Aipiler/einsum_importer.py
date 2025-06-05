from iree.compiler.ir import (
    AffineAddExpr,
    AffineConstantExpr,
    AffineExpr,
    AffineMap,
    AffineMapAttr,
    AffineModExpr,
    AffineMulExpr,
    AffineSymbolExpr,
    Attribute,
    Block,
    Context,
    DenseElementsAttr,
    DenseResourceElementsAttr,
    FloatAttr,
    BF16Type,
    ComplexType,
    Float8E5M2Type,
    Float8E4M3FNType,
    Float8E5M2FNUZType,
    Float8E4M3FNUZType,
    F16Type,
    F32Type,
    F64Type,
    FunctionType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    RankedTensorType,
    Location,
    Module,
    Operation,
    StringAttr,
    SymbolTable,
    Type as IrType,
    UnitAttr,
    Value,
    ShapedType,
)


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
from Aipiler import datatype as dtypes
from iree.compiler import ir
from iree.compiler.dialects.linalg.opdsl.lang import *
from iree.compiler.dialects import builtin
from iree.compiler.dialects import func
from iree.compiler.dialects import linalg
from iree.compiler.dialects import tensor
from Aipiler.aot.support.ir_utils import ModuleBuilder
from Aipiler.graph import EinsumGraph


class Einsum_importer:

    def __init__(
        self,
        module_builder: ModuleBuilder,
    ) -> None:
        self.visited_nodes: List[EinsumPrimitive] = []
        self.symbol_table: Dict[Tensor, ir.Value] = {}
        self.module_builder: ModuleBuilder = module_builder

        self._AIPILER_TO_MLIR: Dict[dtypes.DataType, Callable[[], IrType]] = {
            dtypes.f32: lambda: F32Type.get(),
            dtypes.boolean: lambda: IntegerType.get_signless(1),
        }

        # self._MLIR_TO_AIPILER: Dict[str, dtypes.DataType] = {
        #     F32Type.get(): dtypes.f32,
        #     IntegerType.get_signless(1): dtypes.boolean,
        # }

    def from_dtype(self, dtype: dtypes.DataType) -> IrType:
        if dtype not in self._AIPILER_TO_MLIR:
            raise RuntimeError("Unsupported data type: {} now".format(dtype.name))
        return self._AIPILER_TO_MLIR[dtype]()

    # def to_dtype(self, mlirty: str) -> dtypes.DataType:
    #     if mlirty not in self._MLIR_TO_AIPILER:
    #         raise RuntimeError("Unsupported data type: {} now".format(mlirty))
    #     return self._MLIR_TO_AIPILER[mlirty]

    def import_MapPrimitive(
        self,
        node: MapPrimitive,
    ) -> ir.Value:
        self.visited_nodes.append(node)

        # 从符号表中找到输入张量的value
        input_tensors = node.inputs
        first_value = self.symbol_table[input_tensors[0]]
        second_value = self.symbol_table[input_tensors[1]]
        if first_value is None or second_value is None:
            raise ValueError(
                f"Input tensor {input_tensors[0]} or {input_tensors[1]} not found in symbol table."
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
            domain(*(domain_defs[s] for s in node.iteration_scripts))
            output_indices = tuple(domain_defs[s] for s in node.output_scripts)
            lhs_indices = tuple(domain_defs[s] for s in node.lhs_scripts)
            rhs_indices = tuple(domain_defs[s] for s in node.rhs_scripts)
            # TODO: 当前只支持加减乘数,不能写死
            C[output_indices] = A[lhs_indices] * B[rhs_indices]

        mlir_dtype = self.from_dtype(node.output.dtype)
        shape_list = []
        for d in node.output.symbolic_shapes:
            if d.is_dynamic:
                shape = ShapedType.get_dynamic_size()  # TODO: 这里应该是一个tensor.dim
            else:
                shape = d.get_size()
            shape_list.append(shape)
        # print(f"shape_list: {shape_list}")
        init_result = tensor.empty(shape_list, mlir_dtype)
        op = _map(
            first_value,
            second_value,
            outs=[init_result],
        )

        return op

    def import_ReducePrimitive(self, node: ReducePrimitive) -> ir.Value:
        self.visited_nodes.append(node)

        # 从符号表中找到输入张量的value
        input_tensors = node.inputs
        input_value = self.symbol_table[input_tensors[0]]
        if input_value is None:
            raise ValueError(
                f"Input tensor {input_tensors[0]} not found in symbol table."
            )

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

        mlir_dtype = self.from_dtype(node.output.dtype)
        shape_list = []
        for s in node.output.symbolic_shapes:
            if s.is_dynamic:
                shape = ShapedType.get_dynamic_size()
            else:
                shape = s.size
            shape_list.append(shape)
        init_result = tensor.empty(shape_list, mlir_dtype)
        op = _map(
            input_value,
            outs=[init_result],
        )

        return op

    def import_PopulatePrimitive(self, node: PopulatePrimitive) -> ir.Value:
        self.visited_nodes.append(node)
        return node.output

    def import_UnaryPrimitive(self, node: UnaryPrimitive) -> ir.Value:
        self.visited_nodes.append(node)
        return node.output

    def import_program(
        self,
        graph: EinsumGraph,
        *,
        func_name: str = "main",
        func_visibility: Optional[str] = None,
        import_symbolic_shape_expressions: bool = False,
    ) -> Operation:

        with self.module_builder.context as ctx, Location.unknown():
            with self.module_builder.ip:
                arguments = []
                for inpur_tensor in graph.inputs:
                    shape_list = []
                    for d in inpur_tensor.symbolic_shapes:
                        if d.is_dynamic:
                            shape = ShapedType.get_dynamic_size()
                        else:
                            shape = d.get_size()
                        shape_list.append(shape)
                    mlir_dtype = self.from_dtype(inpur_tensor.dtype)
                    tensor_arg = RankedTensorType.get(shape_list, mlir_dtype)
                    arguments.append(tensor_arg)

                @func.FuncOp.from_py_func(*arguments, name=func_name)
                def generated_func(*args):
                    # 建立输入参数的符号表
                    args_list = list(args)
                    for i, arg in enumerate(args_list):
                        self.symbol_table[graph.inputs[i]] = arg

                    # 遍历所有节点，生成对应的 MLIR 操作
                    for node in graph.nodes:
                        if isinstance(node, MapPrimitive):
                            op_ret: (
                                ir.Operation | ir.Value | tuple | List | ir.OpResult
                            ) = self.import_MapPrimitive(node)
                        elif isinstance(node, ReducePrimitive):
                            op_ret: (
                                ir.Operation | ir.Value | tuple | List | ir.OpResult
                            ) = self.import_ReducePrimitive(node)
                        else:
                            pass

                        if isinstance(op_ret, tuple | List):
                            for i, operation in enumerate(op_ret):
                                if isinstance(operation, ir.Operation) or isinstance(
                                    operation, ir.OpView
                                ):
                                    self.symbol_table[node.output] = operation.result
                                elif isinstance(operation, ir.OpResult):
                                    self.symbol_table[node.output] = operation
                                else:
                                    raise NotImplementedError
                        elif isinstance(op_ret, ir.OpResult):
                            self.symbol_table[node.output] = op_ret
                        else:
                            for i, result in enumerate(op_ret.results):
                                self.symbol_table[node.output] = result

                    # 获得函数的所有输出
                    outputs = (self.symbol_table.get(out) for out in graph.outputs)
                    return outputs

            return generated_func.func_op
