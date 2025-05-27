from typing import List, Dict, Any, Optional, Set, Tuple, Union, Type, Sequence
from Aipiler.tensor import Tensor
from Aipiler.primitive import EinsumPrimitive, MapPrimitive, ReducePrimitive
from Aipiler.visitor import MLIRCodeGenVisitor
from mlir import ir


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
        self._symbol_table = {}


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
            for arg in self.inputs:
                shape_list = list(arg.shape)
                dtype = arg.dtype
                mlir_dtype = self._str_to_mlir_dtype(dtype)
                tensor_arg = ir.RankedTensorType.get(shape_list, mlir_dtype)
                arguments.append(tensor_arg)
            extern_func = []

        # 遍历所有节点，生成对应的 MLIR 操作
        for i, node in enumerate(self.nodes):
            pass

    def _gen_tensor(self, tensor: Tensor) -> ir.Value:
        """生成对应的 MLIR Tensor"""
        # 这里需要根据 tensor 的具体类型和属性来生成 MLIR Tensor
        # 例如，可以使用 tensor 的 shape 和 dtype 来创建 MLIR Tensor
        pass

    def _gen_primitive(self, prim: EinsumPrimitive) -> ir.Operation:
        """生成对应的 MLIR 操作"""
        # 这里需要根据 prim 的具体类型和属性来生成 MLIR 操作
        # 例如，可以使用 prim 的输入、输出和操作类型来创建 MLIR 操作
        if isinstance(prim, MapPrimitive):
            self.visitor.visit_map(prim)
        elif isinstance(prim, ReducePrimitive):
            self.visitor.visit_reduce(prim)
        else:
            raise NotImplementedError(f"Unsupported primitive type: {type(prim)}")

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
