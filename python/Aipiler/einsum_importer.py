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
    UnaryPrimitive,
)
from Aipiler.tensor import Tensor, FakeData, FakeTensor, FakeScalar, Parameter
from Aipiler import datatype as dtypes
from iree.compiler import ir
from Aipiler.lang import *
from iree.compiler.dialects import builtin
from iree.compiler.dialects import func, arith
from iree.compiler.dialects import linalg
from iree.compiler.dialects import tensor
from Aipiler.aot.support.ir_utils import ModuleBuilder
from Aipiler.graph import EinsumGraph
from Aipiler.basic_operator import operator_registry
from Aipiler.dim import Dim

from Aipiler.support.ir_imports import util_d

import numpy as np


class Einsum_importer:

    def __init__(
        self,
        module_builder: ModuleBuilder,
    ) -> None:
        self.visited_nodes: List[EinsumPrimitive] = []
        self.parameter_table: Dict[str, Parameter] = {}
        self.symbol_table: Dict[FakeData, ir.Value] = {}
        self.module_builder: ModuleBuilder = module_builder
        self.module_op: builtin.ModuleOp = module_builder.module_op

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

    def get_dyn_dim(self, d: Dim):
        # get dynamic dim from known dims
        # TODO: simplify this process, the same procedure appeared in `import_program``
        recorded_dims: List[Dim] = []
        for t in self.symbol_table.keys():
            if isinstance(t, FakeTensor):
                recorded_dims += t.symbolic_shapes
        eq_dim: Dim
        for dim in recorded_dims:
            if self.graph.dims_relations.is_equal(d, dim):
                eq_dim = dim
                break
        assert eq_dim is not None
        if eq_dim.is_dynamic:
            _eq_tensor_mlir_val = self.symbol_table[eq_dim._fake_tensor]
            idx = eq_dim._idx_in_tensor
            idx_mlir_val = arith.constant(
                ir.IndexType.get(), IntegerAttr.get(ir.IndexType.get(), idx)
            )
            shape = tensor.dim(_eq_tensor_mlir_val, idx_mlir_val)
        else:
            # else, create by arith.constant
            shape = eq_dim.size
            assert isinstance(shape, (int, float))
        return shape

    def constant_scalar(self, dtype: dtypes.DataType, scalar: Union[int, float]):
        if isinstance(dtype, dtypes.integer.IntegerType):
            size_attr = IntegerAttr.get(self.from_dtype(dtype), scalar)
        else:
            assert isinstance(dtype, dtypes.float.FloatType)
            size_attr = FloatAttr.get(self.from_dtype(dtype), scalar)

        return arith.constant(self.from_dtype(dtype), size_attr)

    def init_empty_tensor(self, output: FakeTensor, init: Union[int, float] = 0):
        """
        build `tensor.empty` operation from FakeTensor
        """
        from iree.compiler.dialects.linalg.opdsl.ops.core_named_ops import fill

        assert isinstance(output, FakeTensor)
        mlir_dtype = self.from_dtype(output.dtype)
        shape_list = []
        for d in output.symbolic_shapes:
            if d.is_dynamic:
                shape = self.get_dyn_dim(d)
            else:
                shape = d.size
                assert isinstance(shape, (int, float))
            shape_list.append(shape)
        # print(f"shape_list: {shape_list}")
        init_result = tensor.empty(shape_list, mlir_dtype)
        cst = self.constant_scalar(output.dtype, init)
        filled = fill(cst, outs=[init_result])
        return filled

    def import_MapPrimitive(
        self,
        node: MapPrimitive,
    ) -> ir.Value:
        self.visited_nodes.append(node)

        # 根据einsum_str 构建linalg.generic op
        symbol_defs = {}
        domain_defs = {}
        for script in node.iteration_axes:
            symbol_defs[script.name] = getattr(S, script.name)
            domain_defs[script.name] = getattr(D, script.name)

        # 获取map op
        map_op = node.op.get_op_callable()

        @linalg_structured_op
        def _map_tensor_tensor(
            A=TensorDef(T, *(symbol_defs[s.name] for s in node.lhs_axes)),
            B=TensorDef(
                T, *(symbol_defs[s.name] for s in node.rhs_axes if not s.is_scalar)
            ),
            C=TensorDef(
                T,
                *(symbol_defs[s.name] for s in node.output_axes),
                output=True,
            ),
        ):
            domain(*(domain_defs[s.name] for s in node.iteration_axes))
            output_indices = tuple(domain_defs[s.name] for s in node.output_axes)
            lhs_indices = tuple(domain_defs[s.name] for s in node.lhs_axes)
            rhs_indices = tuple(
                domain_defs[s.name] for s in node.rhs_axes if not s.is_scalar
            )
            # TODO: 当前只支持加减乘数,不能写死
            C[output_indices] = map_op(A[lhs_indices], B[rhs_indices])

        @linalg_structured_op
        def _map_tensor_scalar(
            A=TensorDef(T, *(symbol_defs[s.name] for s in node.lhs_axes)),
            B=ScalarDef(T),
            C=TensorDef(
                T,
                *(symbol_defs[s.name] for s in node.output_axes),
                output=True,
            ),
        ):
            domain(*(domain_defs[s.name] for s in node.iteration_axes))
            output_indices = tuple(domain_defs[s.name] for s in node.output_axes)
            lhs_indices = tuple(domain_defs[s.name] for s in node.lhs_axes)
            # TODO: 当前只支持加减乘数,不能写死
            C[output_indices] = map_op(A[lhs_indices], B)

        if all(isinstance(inp, FakeTensor) for inp in node.inputs):
            # 从符号表中找到输入张量的value
            input_tensors = node.inputs
            first_value = self._get_mlir_value(input_tensors[0])
            second_value = self._get_mlir_value(input_tensors[1])
            if first_value is None or second_value is None:
                raise ValueError(
                    f"Input tensor {input_tensors[0]} or {input_tensors[1]} not found in symbol table."
                )

            init_result = self.init_empty_tensor(node.output)
            op = _map_tensor_tensor(
                first_value,
                second_value,
                outs=[init_result],
            )

            return op
        elif all(isinstance(inp, FakeScalar) for inp in node.inputs):
            # scalar op scalar
            # TODO
            raise NotImplementedError("Unsupport for map for scalar.")
        else:
            # scalar operate tensor
            # distinguish scalar and tensor
            _scalar: FakeScalar
            _tensor: FakeTensor
            if isinstance(node.lhs, FakeScalar) and isinstance(node.rhs, FakeTensor):
                _scalar, _tensor = node.lhs, node.rhs
            else:
                _scalar, _tensor = node.rhs, node.lhs

            _tensor_mlir_val = self._get_mlir_value(_tensor)
            _scalar_mlir_value = self._get_mlir_value(_scalar)

            # create init empty tensor from output tensor
            init_result = self.init_empty_tensor(node.output)
            op = _map_tensor_scalar(
                _tensor_mlir_val,
                _scalar_mlir_value,
                outs=[init_result],
            )

            return op

    def import_ReducePrimitive(self, node: ReducePrimitive) -> ir.Value:
        self.visited_nodes.append(node)

        # 从符号表中找到输入张量的value
        input_tensors = node.inputs
        input_value = self._get_mlir_value(input_tensors[0])
        if input_value is None:
            raise ValueError(
                f"Input tensor {input_tensors[0]} not found in symbol table."
            )

        # 根据einsum_str 构建linalg.generic op
        symbol_defs = {}
        domain_defs = {}
        for script in node.iteration_axes:
            symbol_defs[script.name] = getattr(S, script.name)
            domain_defs[script.name] = getattr(D, script.name)

        # 获取reduce op
        reduce_op = ReduceFnType(node.op.get_op_callable())

        @linalg_structured_op
        def _reduce(
            INPUT=TensorDef(T, *(symbol_defs[s.name] for s in node.x_axes)),
            OUTPUT=TensorDef(
                T,
                *(symbol_defs[s.name] for s in node.output_axes),
                output=True,
            ),
        ):
            domain(*(domain_defs[s.name] for s in node.iteration_axes))
            output_indices = tuple(domain_defs[s.name] for s in node.output_axes)
            input_indices = tuple(domain_defs[s.name] for s in node.x_axes)
            target_dim_indices = tuple(domain_defs[s] for s in node.dims_to_reduce)
            # TODO: 当前只支持加减乘数,不能写死
            OUTPUT[output_indices] = reduce_op[target_dim_indices](INPUT[input_indices])

        init_result = self.init_empty_tensor(node.output)
        op = _reduce(
            input_value,
            outs=[init_result],
        )

        return op

    def import_UnaryPrimitive(self, node: UnaryPrimitive) -> ir.Value:
        self.visited_nodes.append(node)

        # 根据einsum_str 构建linalg.generic op
        symbol_defs = {}
        domain_defs = {}
        for script in node.iteration_axes:
            symbol_defs[script.name] = getattr(S, script.name)
            domain_defs[script.name] = getattr(D, script.name)

        # 获取map op
        unary_op = node.op.get_op_callable()

        # 从符号表中找到输入张量的value
        input_tensors = node.inputs
        input_val = self._get_mlir_value(input_tensors[0])
        if input_val is None:
            raise ValueError(
                f"Input tensor {input_tensors[0]} not found in symbol table."
            )
        if isinstance(input_val, FakeScalar):
            raise NotImplementedError()

        @linalg_structured_op
        def elemwise_unary(
            I=TensorDef(T, *(symbol_defs[s.name] for s in node.x_axes)),
            O=TensorDef(
                U, *(symbol_defs[s.name] for s in node.output_axes), output=True
            ),
            fun=UnaryFnAttrDef(default=UnaryFn.sqrt),
            cast=TypeFnAttrDef(default=TypeFn.cast_signed),
        ):
            """Applies the unary function fun elementwise.

            Numeric casting is performed on the input operand, promoting it to the same
            data type as the accumulator/output.
            """
            O[None] = fun(cast(U, I[None]))

        init_result = self.init_empty_tensor(node.output)
        op = elemwise_unary(input_val, outs=[init_result], fun=unary_op)
        return op

    def _get_mlir_value(self, tensor: FakeTensor):
        if tensor in self.symbol_table:
            return self.symbol_table[tensor]
        assert isinstance(tensor, Parameter), f"Can not find {tensor}."
        mlir_value = self._lift_tensor_to_global(tensor)
        self.symbol_table[tensor] = mlir_value
        return mlir_value

    def _lift_tensor_to_global(self, literal: Parameter) -> ir.Value:
        """lift tensor to module attribute and global declare
        Args:
            literal (Parameter): _tensor literal_

        Returns:
            ir.Value: _global operation_
        """
        name = "parameter_{}".format(len(self.parameter_table))
        self.parameter_table[name] = literal
        with InsertionPoint.at_block_begin(
            self.module_op.regions[0].blocks[0]
        ), Location.unknown():
            element_type = self.from_dtype(literal.dtype)
            tensor_type = RankedTensorType.get(literal.numeric_shape, element_type)
            ir_attrs = {
                "sym_name": StringAttr.get(name),
                "sym_visibility": StringAttr.get("private"),
                "type": ir.TypeAttr.get(tensor_type),
                "noinline": UnitAttr.get(),
                # TODO: "device":
            }

            detached_tensor = literal.storage.detach().contiguous().cpu().numpy()
            content = memoryview(detached_tensor)
            elements_attr = DenseResourceElementsAttr.get_from_buffer(
                content, name, tensor_type
            )
            ir_attrs["initial_value"] = elements_attr
            Operation.create("util.global", attributes=ir_attrs)
        loaded_value = util_d.GlobalLoadOp(tensor_type, name).result
        return loaded_value

    def import_program(
        self,
        graph: EinsumGraph,
        *,
        func_name: str = "main",
        func_visibility: Optional[str] = None,
        import_symbolic_shape_expressions: bool = False,
    ) -> Operation:
        self.graph = graph
        with self.module_builder.context as ctx, Location.unknown():
            with self.module_builder.ip:
                # get function input types
                input_tensor_args: List[FakeTensor] = []
                scalar_args: List[FakeScalar] = []
                param_args: List[Parameter] = []
                # collect from args
                for input_tensor in graph.inputs:
                    if isinstance(input_tensor, FakeTensor):
                        if isinstance(input_tensor, Parameter):
                            param_args.append(input_tensor)
                        else:
                            input_tensor_args.append(input_tensor)
                    else:
                        # scalar are not argument of function
                        assert isinstance(input_tensor, FakeScalar)
                        scalar_args.append(input_tensor)

                # get function argument types
                function_argument_types = []
                for input_tensor in input_tensor_args:
                    shape_list = []
                    for d in input_tensor.symbolic_shapes:
                        if d.is_dynamic:
                            # kdynamic if dim is dynamic
                            shape = ShapedType.get_dynamic_size()
                        else:
                            shape = d.size
                        shape_list.append(shape)
                    mlir_dtype = self.from_dtype(input_tensor.dtype)
                    tensor_arg_type = RankedTensorType.get(shape_list, mlir_dtype)
                    function_argument_types.append(tensor_arg_type)

                @func.FuncOp.from_py_func(*function_argument_types, name=func_name)
                def generated_func(*args):
                    # initialize symbol table
                    # function input tensors
                    args_list = list(args)
                    assert len(args_list) == len(input_tensor_args)
                    for input_tensor, arg in zip(input_tensor_args, args_list):
                        # function input arg --> mlir value
                        self.symbol_table[input_tensor] = arg

                    # for scalars
                    # init scalars at first of all
                    for scalar in scalar_args:
                        scalar_mlir_value: Value
                        if isinstance(scalar.sym_val, Dim):
                            # search equivalent dim and its fake tensor
                            _eq_dim: Dim
                            _eq_tensor: FakeTensor
                            if scalar.sym_val._fake_tensor in self.symbol_table:
                                _eq_dim = scalar.sym_val
                                _eq_tensor = scalar.sym_val._fake_tensor
                            else:
                                # get equivalent dim from disjoint set
                                for input_tensor in input_tensor_args:
                                    for dim in input_tensor.symbolic_shapes:
                                        if graph.dims_relations.is_equal(
                                            scalar.sym_val, dim
                                        ):
                                            _eq_dim = dim
                                            _eq_tensor = input_tensor
                                            break
                            assert _eq_dim is not None
                            assert _eq_tensor is not None

                            # if _eq_dim is dynamic, get by op: tensor.dim
                            if _eq_dim.is_dynamic:
                                _eq_tensor_mlir_val = self.symbol_table[_eq_tensor]
                                idx = _eq_dim._idx_in_tensor
                                idx_mlir_val = arith.constant(
                                    ir.IndexType.get(),
                                    IntegerAttr.get(ir.IndexType.get(), idx),
                                )
                                _scalar_mlir_val_index = tensor.dim(
                                    _eq_tensor_mlir_val, idx_mlir_val
                                )

                                if isinstance(scalar.dtype, dtypes.integer.IntegerType):
                                    scalar_mlir_value = arith.index_cast(
                                        self.from_dtype(scalar.dtype),
                                        _scalar_mlir_val_index,
                                    )
                                elif isinstance(scalar.dtype, dtypes.float.FloatType):
                                    _scalar_mlir_value_i32 = arith.index_cast(
                                        IntegerType.get_signless(32),
                                        _scalar_mlir_val_index,
                                    )
                                    scalar_mlir_value = arith.sitofp(
                                        self.from_dtype(scalar.dtype),
                                        _scalar_mlir_value_i32,
                                    )
                                else:
                                    raise NotImplementedError()
                            else:
                                # else, create by arith.constant
                                size = _eq_dim.size
                                assert isinstance(size, (int, float))
                                if isinstance(scalar.dtype, dtypes.integer.IntegerType):
                                    size_attr = IntegerAttr.get(
                                        self.from_dtype(scalar.dtype), size
                                    )
                                else:
                                    assert isinstance(
                                        scalar.dtype, dtypes.float.FloatType
                                    )
                                    size_attr = FloatAttr.get(
                                        self.from_dtype(scalar.dtype), size
                                    )

                                scalar_mlir_value = arith.constant(
                                    self.from_dtype(scalar.dtype), size_attr
                                )

                        else:
                            assert isinstance(scalar.sym_val, (int, float))
                            if isinstance(scalar.dtype, dtypes.integer.IntegerType):
                                sym_attr = IntegerAttr.get(
                                    self.from_dtype(scalar.dtype), scalar.sym_val
                                )
                            else:
                                assert isinstance(scalar.dtype, dtypes.float.FloatType)
                                sym_attr = FloatAttr.get(
                                    self.from_dtype(scalar.dtype), scalar.sym_val
                                )

                            scalar_mlir_value = arith.constant(
                                self.from_dtype(scalar.dtype), sym_attr
                            )
                        # insert into symbol table
                        self.symbol_table[scalar] = scalar_mlir_value

                    # for parameters
                    # for param in param_args:
                    #     param_mlir_value = self._lift_tensor_to_global(param)
                    #     self.symbol_table[param] = param_mlir_value

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
                        elif isinstance(node, UnaryPrimitive):
                            op_ret: (
                                ir.Operation | ir.Value | tuple | List | ir.OpResult
                            ) = self.import_UnaryPrimitive(node)
                        else:
                            raise NotImplementedError()

                        if isinstance(op_ret, Sequence):
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
