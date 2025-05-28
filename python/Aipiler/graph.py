from typing import List, Dict, Any, Optional, Set, Tuple, Union, Type, Sequence
from Aipiler.tensor import Tensor
from Aipiler.primitive import EinsumPrimitive, MapPrimitive, ReducePrimitive
from Aipiler.visitor import MLIRCodeGenVisitor
from Aipiler.datatype import DtypeMapper
from mlir.dialects import arith, builtin, func, linalg, tensor
from mlir.dialects.linalg.opdsl.lang import *
from mlir.ir import *


class EinsumGraph:
    """节点管理器，负责管理计算图中的所有节点"""

    def __init__(
        self, outputs: Sequence[Tensor], inputs: Optional[Sequence[Tensor]] = None
    ):
        self.outputs = list(outputs)
        self.inputs: Optional[List[Tensor]] = list(inputs) if inputs else []
        self.nodes: List[EinsumPrimitive] = self.update_nodes()
        self._mlir_context = ir.Context()
        self._module = ir.Module.create()
        self._func_name = "main"
        self._symbol_table: Dict[Tensor, Value] = {}
        self.visitor = MLIRCodeGenVisitor(self._mlir_context, self._symbol_table)

    def update_nodes(self):
        nodes: List[EinsumPrimitive] = []
        stack: List[EinsumPrimitive] = [output._trace for output in self.outputs]
        while stack:
            op = stack.pop()
            nodes.insert(0, op)
            for i in op.inputs:
                if i._trace:
                    stack.append(i._trace)
                else:
                    if i not in self.inputs:
                        self.inputs.append(i)
        return nodes

    def codegen(self):
        """生成 MLIR 代码"""

        # 遍历所有输入张量，生成对应的 MLIR Tensor
        for input_tensor in self.inputs:
            pass

        with ir.InsertionPoint(self._module.body):
            arguments = []
            for inpur_tensor in self.inputs:
                # TODO：暂时使用tensor的shape，后期再使用symbolic_shape
                shape_list = list(inpur_tensor.shape)
                mlir_dtype = DtypeMapper.to_mlir(input_tensor.dtype)
                tensor_arg = ir.RankedTensorType.get(shape_list, mlir_dtype)
                arguments.append(tensor_arg)

            @func.FuncOp.from_py_func(*arguments, name=self._func_name)
            def generated_func(*args):
                # 建立输入参数的符号表
                args_list = list(args)
                for i, arg in enumerate(args_list):
                    self._symbol_table[self.inputs[i]] = arg

                # 遍历所有节点，生成对应的 MLIR 操作
                for node in self.nodes:
                    op_ret: ir.Operation | ir.Value | tuple | List | ir.OpResult = (
                        node.accept(self.visitor)
                    )

                    if isinstance(op_ret, tuple | List):
                        for i, operation in enumerate(op_ret):
                            if isinstance(operation, ir.Operation) or isinstance(
                                operation, ir.OpView
                            ):
                                self._symbol_table[node.output] = operation.result
                            elif isinstance(operation, ir.OpResult):
                                self._symbol_table[node.output] = operation
                            else:
                                raise NotImplementedError
                    elif isinstance(op_ret, ir.OpResult):
                        self._symbol_table[node.output] = op_ret
                    else:
                        for i, result in enumerate(op_ret.results):
                            self._symbol_table[node.output] = result

                # 获得函数的所有输出
                outputs = (self._symbol_table.get(out) for out in self.outputs)
                return outputs

        print(self._module)

    def __str__(self) -> str:
        tensors = []

        def nameof(t):
            if t not in tensors:
                assert False
            return "t" + str(tensors.index(t))

        doc = "Graph("
        tensors += self.inputs
        param_doc = []
        for inp in self.inputs:
            n = nameof(inp)
            param_doc.append(n)
        doc += ", ".join(param_doc)
        doc += ")\n"

        for prim in self.nodes:
            if isinstance(prim, MapPrimitive):
                lhs = prim.inputs[0]
                rhs = prim.inputs[1]
                ret = prim.output
                tensors.append(ret)
                prim_doc = '{ret} = map({lhs}, {rhs}, "{einsum_str}", [{map_dims}], "{op}")'.format(
                    ret=nameof(ret),
                    lhs=nameof(lhs),
                    rhs=nameof(rhs),
                    einsum_str=prim.einsum_str,
                    map_dims=", ".join(
                        ['"{}"'.format(letter) for letter in prim.ranks_to_map]
                    ),
                    op=prim.op.name,
                )
                doc += "\t"
                doc += prim_doc
            elif isinstance(prim, ReducePrimitive):
                inp = prim.inputs[0]
                ret = prim.output
                tensors.append(ret)
                prim_doc = '{ret} = reduce({inp}, "{einsum_str}", "{reduce_dims}", "{op}")'.format(
                    ret=nameof(ret),
                    inp=nameof(inp),
                    einsum_str=prim.einsum_str,
                    reduce_dims=prim.reduce_rank,
                    op=prim.op.name,
                )
                doc += "\t"
                doc += prim_doc
            else:
                doc += prim.__str__
            doc += "\n"
        doc += "\treturn "
        outputs = [nameof(out) for out in self.outputs]
        doc += ", ".join(outputs)
        return doc
