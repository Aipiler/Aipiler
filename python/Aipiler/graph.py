from typing import List, Dict, Any, Optional, Set, Tuple, Union, Type, Sequence
from Aipiler.tensor import FakeTensor, FakeScalar, FakeData, Parameter
from Aipiler.primitive import (
    EinsumPrimitive,
    MapPrimitive,
    ReducePrimitive,
    UnaryPrimitive,
    CascadePrimitive,
)
from Aipiler.dim import Dim, DisjointSetUnion

EINSUM_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class DimDimRelation:
    def __init__(self):
        self._eq_dict: Dict[Dim, Set[Dim]] = dict()
        self._depend_eq_dict_key_input: Dict[Dim, Set[Dim]] = dict()
        self._depend_eq_dict_key_output: Dict[Dim, Set[Dim]] = dict()

    def is_equal(self, d0: Dim, d1: Dim):
        if d0 not in self._eq_dict:
            raise ValueError(f"Cannot find dim: {d0}, is it a wild dim?")
        if d1 not in self._eq_dict:
            raise ValueError(f"Cannot find dim: {d1}, is it a wild dim?")
        return d1 in self._eq_dict[d0]

    def __repr__(self):
        from Aipiler.utils.printer import P

        with P.section("Dims Relations(Cross Prims)"):
            with P.section("EQ"):
                if len(self._eq_dict) == 0:
                    P.add_line("Nothing")
                else:
                    with P.table(
                        separator=" | ", aligns=["c", "c"], col_widths=[30, 80]
                    ) as t:
                        t.add_row("Dim", "Equal To")
                        t.add_row("-" * 30, "-" * 80)
                        for d, s in self._eq_dict.items():
                            t.add_row(str(d), str(s))
            with P.section("DEPEND&EQ"):
                if len(self._depend_eq_dict_key_input) == 0:
                    P.add_line("Nothing")
                else:
                    with P.table(
                        separator=" | ", aligns=["c", "c"], col_widths=[30, 80]
                    ) as t:
                        t.add_row("Input Dim", "Depend and Equal By")
                        t.add_row("-" * 30, "-" * 80)
                        for d, s in self._depend_eq_dict_key_input.items():
                            t.add_row(str(d), str(s))
            P.add_line()
        return str(P)


class EinsumGraph:
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
        self.dims_relations = DimDimRelation()

    def update_nodes(self) -> "EinsumGraph":
        nodes: List[EinsumPrimitive] = []
        stack: List[EinsumPrimitive] = [output._trace for output in self.outputs]
        while stack:
            op = stack.pop()
            nodes.insert(0, op)
            for i in op.inputs:
                if i in self.inputs:
                    # if i in self.inputs, stop tracing forward
                    continue
                if isinstance(i, FakeTensor):
                    if isinstance(i, Parameter) and i not in self.inputs:
                        self.inputs.append(i)
                    else:
                        assert i._trace is not None
                        stack.append(i._trace)
                else:
                    assert isinstance(i, FakeScalar)
                    self.inputs.append(i)

        self.nodes = nodes
        self.update_dims_relations()
        return self

    def update_dims_relations(self):
        for node in self.nodes:
            # insert eq dims
            equal_dict = self.dims_relations._eq_dict
            for axis, eq_set in node.axes_relations.equal_dict.items():
                d = node.dim_axis_relations[axis]
                dims_eq_set = set([node.dim_axis_relations[a] for a in eq_set])
                if d in equal_dict:
                    equal_dict[d].update(dims_eq_set)
                    dims_eq_set = equal_dict[d]
                else:
                    equal_dict[d] = dims_eq_set
                # equal dims map to the same set
                # update other sets
                for _d in dims_eq_set:
                    if _d in equal_dict:
                        equal_dict[_d].update(dims_eq_set)
                    else:
                        equal_dict[_d] = dims_eq_set

            # insert depend eq dim
            # input as key
            depend_eq_dict_key_input = self.dims_relations._depend_eq_dict_key_input
            for axis, eq_set in node.axes_relations.depend_eq_dict_key_input.items():
                d = node.dim_axis_relations[axis]
                dims_eq_set = set([node.dim_axis_relations[a] for a in eq_set])
                if d in depend_eq_dict_key_input:
                    depend_eq_dict_key_input[d].update(dims_eq_set)
                else:
                    depend_eq_dict_key_input.update({d: dims_eq_set})
            # output as key
            depend_eq_dict_key_output = self.dims_relations._depend_eq_dict_key_output
            for axis, eq_set in node.axes_relations.depend_eq_dict_key_output.items():
                d = node.dim_axis_relations[axis]
                dims_eq_set = set([node.dim_axis_relations[a] for a in eq_set])
                if d in depend_eq_dict_key_output:
                    depend_eq_dict_key_output[d].update(dims_eq_set)
                else:
                    depend_eq_dict_key_output.update({d: dims_eq_set})
        # handle long dependence
        for i_d, o_dims in self.dims_relations._depend_eq_dict_key_input.items():
            long_depends = set()
            for o_d in o_dims:
                # if o_d is input of another primitive
                if o_d in self.dims_relations._depend_eq_dict_key_input:
                    long_depends.update(
                        self.dims_relations._depend_eq_dict_key_input[o_d]
                    )
                    self.dims_relations._depend_eq_dict_key_output[o_d].add(i_d)
            o_dims.update(long_depends)

    def __repr__(self):
        from Aipiler.utils import printer, namer

        input_names = [namer.N.get_or_create_name_of(i) for i in self.inputs]
        with printer.P.section("Graph({})".format(", ".join(input_names))):
            for node in self.nodes:
                if isinstance(node, MapPrimitive):
                    printer.P.add_line(
                        '{ret} = map({lhs}, {rhs}, "{einsum_str}", "{op}")\t\t# {name}'.format(
                            ret=namer.N.get_or_create_name_of(node.output),
                            lhs=namer.N.get_or_create_name_of(node.lhs),
                            rhs=namer.N.get_or_create_name_of(node.rhs),
                            einsum_str=node.einsum_str,
                            op=node.op.name,
                            name=namer.N.get_or_create_name_of(node),
                        )
                    )
                elif isinstance(node, ReducePrimitive):
                    printer.P.add_line(
                        '{ret} = reduce({x}, "{einsum_str}", "{reduce_dims}", "{op}")\t\t# {name}'.format(
                            ret=namer.N.get_or_create_name_of(node.output),
                            x=namer.N.get_or_create_name_of(node.x),
                            einsum_str=node.einsum_str,
                            reduce_dims=", ".join(node.dims_to_reduce),
                            op=node.op.name,
                            name=namer.N.get_or_create_name_of(node),
                        )
                    )
                elif isinstance(node, UnaryPrimitive):
                    printer.P.add_line(
                        '{ret} = unary({x}, "{einsum_str}", "{op}"\t\t# {name})'.format(
                            ret=namer.N.get_or_create_name_of(node.output),
                            x=namer.N.get_or_create_name_of(node.x),
                            einsum_str=node.einsum_str,
                            op=node.op.name,
                            name=namer.N.get_or_create_name_of(node),
                        )
                    )
                else:
                    printer.P.add_line("UnImplemented stringify.")
            printer.P.add_line(
                "return {}".format(
                    ", ".join([namer.N.get_or_create_name_of(o) for o in self.outputs])
                )
            )
            printer.P.add_line()
            with printer.P.section("Nodes in Graph:"):
                for node in self.nodes:
                    printer.P.add_line(str(node))
                    printer.P.add_line()

            printer.P.add_line()
            printer.P.add_line(str(self.dims_relations))
            printer.P.add_line()
            return str(printer.P)

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


def trace_from(
    outputs: Optional[Union[FakeTensor, Sequence[FakeTensor]]],
    inputs: Optional[Union[FakeTensor, Sequence[FakeTensor]]] = None,
) -> EinsumGraph:
    """
    TODO: trace einsumgraph with `inputs` and `outputs`. If inputs are None, trace until leaves of whole graph.
    for example:
    the whole graph is: t1=reduce(t0=map(A, B, ...), ...);
    if inputs = [t0], outputs = [t1], then the result of trace_from has nodes: [reduce]
    elif inputs= [A, B], outputs= [t1], then the result of trace_from has nodes: [reduce, map]
    """
    if isinstance(outputs, FakeTensor):
        if outputs._trace is None:
            raise ValueError("trace_from expects symbol tensor(s)")
        outputs = [outputs]
    else:
        outputs = list(outputs)
        assert all(isinstance(v, FakeTensor) for v in outputs)

    if inputs is not None:
        if isinstance(inputs, FakeTensor):
            inputs = [inputs]
        else:
            inputs = list(inputs)
    return EinsumGraph(outputs, inputs).update_nodes()
