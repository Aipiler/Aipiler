from typing import List, Dict, Any, Optional, Set, Tuple, Union, Type, Sequence
from Aipiler.tensor import FakeTensor, FakeScalar, FakeData
from Aipiler.primitive import (
    EinsumPrimitive,
    MapPrimitive,
    ReducePrimitive,
    UnaryPrimitive,
    CascadePrimitive,
)
from Aipiler.visitor import MLIRCodeGenVisitor
from Aipiler.dim import Dim, DisjointSetUnion

EINSUM_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class EinsumGraph:
    """节点管理器，负责管理计算图中的所有节点"""

    def __init__(
        self,
        outputs: Sequence[FakeData],
        inputs: Sequence[FakeData],
        name: str = "main",
    ):
        self.name = name
        self.outputs = list(outputs)
        self.inputs = list(inputs)
        self.nodes: List[EinsumPrimitive]
        self.sym_dim_set: DisjointSetUnion

    def update_nodes(self) -> "EinsumGraph":
        nodes: List[EinsumPrimitive] = []
        stack: List[EinsumPrimitive] = [output._trace for output in self.outputs]
        while stack:
            op = stack.pop()
            nodes.insert(0, op)
            for i in op.inputs:
                if i in self.inputs:
                    continue
                if isinstance(i, FakeTensor):
                    assert i._trace is not None
                    stack.append(i._trace)
                else:
                    assert isinstance(i, FakeScalar)
                    self.inputs.append(i)

        self.nodes = nodes
        self.sym_dim_set = self.update_dim_value_set()
        return self

    def update_dim_value_set(self):
        """
        更新图中所有节点的维度值集合
        """
        sym_dim_set = DisjointSetUnion()
        for node in self.nodes:
            inputs_scripts = node.inputs_scripts
            outputs_scripts = node.outputs_scripts

            input_tensors = node.inputs
            output_tensor = node.outputs

            idx_dim_dict: Dict[str, List[Dim]] = {}
            for input_script, input_tensor in zip(inputs_scripts, input_tensors):
                if not input_script:  # this input is scalar
                    continue
                assert isinstance(input_tensor, FakeTensor)
                for input_idx, input_dim in zip(
                    input_script, input_tensor.symbolic_shapes
                ):
                    if input_idx not in idx_dim_dict:
                        idx_dim_dict[input_idx] = []
                    idx_dim_dict[input_idx].append(input_dim)

            for output_script, output_tensor in zip(outputs_scripts, output_tensor):
                assert isinstance(output_tensor, FakeTensor)
                for output_idx, output_dim in zip(
                    output_script, output_tensor.symbolic_shapes
                ):
                    if output_idx not in idx_dim_dict:
                        idx_dim_dict[output_idx] = []
                    idx_dim_dict[output_idx].append(output_dim)

            # 更新维度值集合
            for script, dim_list in idx_dim_dict.items():
                sym_dim_set.union(*dim_list)

        for value_dim_set in sym_dim_set.get_all_value_dim_set():
            value_dim_set.populate_dim_size()
        return sym_dim_set

    def __str__(self) -> str:
        tensors = []
        cascade_docs = []

        def nameof(t):
            if isinstance(t, FakeScalar):
                if isinstance(t.sym_val, Dim):
                    # find
                    src_tensor_name = nameof(t.sym_val._fake_tensor)
                    dim_idx = t.sym_val._idx_in_tensor
                    return f"{src_tensor_name}.dim{dim_idx}"
                else:
                    return f"{t.sym_val}"
            else:
                if t not in tensors:
                    assert False
                return "t" + str(tensors.index(t))

        doc = "Graph("
        tensors += self.inputs
        param_doc = []
        input_tensors = [t for t in self.inputs if isinstance(t, FakeTensor)]
        input_scalars = [t for t in self.inputs if isinstance(t, FakeScalar)]
        # print tensor first because of scalar maybe dim
        for inp in input_tensors:
            n = nameof(inp)
            param_doc.append(n)
        for inp in input_scalars:
            n = nameof(inp)
            param_doc.append(n)
        doc += ", ".join(param_doc)
        doc += ")\n"

        # 打印graph
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
                        ['"{}"'.format(letter) for letter in prim.dims_to_map]
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
                    reduce_dims=prim.dims_to_reduce,
                    op=prim.op.name,
                )
                doc += "\t"
                doc += prim_doc
            elif isinstance(prim, UnaryPrimitive):
                inp = prim.inputs[0]
                ret = prim.output
                tensors.append(ret)
                prim_doc = '{ret} = unary({inp}, "{einsum_str}", "{op}")'.format(
                    ret=nameof(ret),
                    inp=nameof(inp),
                    einsum_str=prim.einsum_str,
                    op=prim.op.name,
                )
                doc += "\t"
                doc += prim_doc
            elif isinstance(prim, CascadePrimitive):
                for ret in prim.outputs:
                    tensors.append(ret)
                prim_doc = '{ret} = cascade{i}({inp}, "{einsum_str}")'.format(
                    ret=", ".join([nameof(ret) for ret in prim.outputs]),
                    i=len(cascade_docs),
                    inp=", ".join([nameof(inp) for inp in prim.inputs]),
                    einsum_str=prim.einsum_str,
                )
                cascade_docs.append(str(prim.graph))
                doc += "\t"
                doc += prim_doc
            else:
                doc += "Unstringify Primitive: " + prim.__class__.__name__
            doc += "\n"
        doc += "\treturn "
        outputs = [nameof(out) for out in self.outputs]
        doc += ", ".join(outputs)
        doc += "\n"

        # 打印value并查集
        doc += "\nSymbolic Dim Set(\n"
        for value_dim_set in set(self.sym_dim_set.dim_set_dict.values()):
            dim_name_list = []
            for dim in value_dim_set.dim_set:
                tensor_name = nameof(dim.fake_tensor)
                dim_idx = dim.index_in_tensor
                dim_name = f"{tensor_name}.dim{dim_idx}"
                dim_name_list.append(dim_name)
            doc += "\t({}),\n".format(", ".join(dim_name_list))
        doc += ")\n\n"

        for i, c_doc in enumerate(cascade_docs):
            doc += "cascade{}:\n".format(i)
            doc += c_doc
        return doc

    def summary_einsum_str(self):
        """
        summarize einsum str from graph
        for example:
            0: T = map(A, B, "ik, kj -> ikj", ...),
            1: C = reduce(T, "ikj -> ij")
        einsum of graph: "ik, kj -> ij"
        """
        from Aipiler.dim import ValueDimSet

        dim_map: Dict[ValueDimSet, str] = {}
        dim_set_dict_set = set(self.sym_dim_set.dim_set_dict.values())
        assert len(dim_set_dict_set) < len(EINSUM_ALPHABET)
        for i, dim_set_dict in enumerate(dim_set_dict_set):
            dim_map[dim_set_dict] = EINSUM_ALPHABET[i]

        inp_strs = []
        for inp in self.inputs:
            inp_str = ""
            if isinstance(inp, FakeScalar):
                inp_str = "_"
            else:
                assert isinstance(inp, FakeTensor)
                for d in inp.symbolic_shapes:
                    dim_set_of_d = self.sym_dim_set.find(d)
                    assert dim_set_of_d in dim_map
                    inp_str += dim_map[dim_set_of_d]
            inp_strs.append(inp_str)

        out_strs = []
        for out in self.outputs:
            out_str = ""
            if isinstance(out, FakeScalar):
                out_str = "_"
            else:
                assert isinstance(out, FakeTensor)
                for d in out.symbolic_shapes:
                    dim_set_of_d = self.sym_dim_set.find(d)
                    assert dim_set_of_d in dim_map
                    out_str += dim_map[dim_set_of_d]
            out_strs.append(out_str)

        return "{inputs_scripts}->{outputs_scripts}".format(
            inputs_scripts=",".join(inp_strs), outputs_scripts=", ".join(out_strs)
        )
