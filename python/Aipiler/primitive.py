from Aipiler.tensor import FakeTensor, FakeData, FakeScalar
from Aipiler.basic_operator import ComputeOperator
from Aipiler.dim import Dim, dims
from typing import List, Union, Sequence, Dict, Any, overload, Callable, Tuple
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from Aipiler.utils import parse_einsum_str
from copy import copy


class EinsumPrimitive(ABC):
    def __init__(self, inputs: List[FakeData], einsum_str: str) -> None:
        self.inputs = inputs
        self.einsum_str = einsum_str
        self.output: Union[FakeData, Sequence[FakeData]] = None
        self.input_scripts, self.output_scripts = parse_einsum_str(self.einsum_str)
        # update scripts
        for scripts in (*self.input_scripts, self.output_scripts):
            if scripts[0] == "_" and len(scripts) == 1:
                scripts.clear()
        # iter scripts
        _ = []
        for sp in self.input_scripts:
            _ += sp
        _ += self.output_scripts
        self.iteration_scripts = set(_)

    def run(self):
        """
        check inputs and einsum, generate symbolic outputs
        """
        fake_tensor_shape: List[Dim] = dims(self.output_scripts)
        dtype = self.inputs[0].dtype
        return FakeTensor(symbolic_shapes=fake_tensor_shape, dtype=dtype, trace=self)

    def accept(self, visitor) -> None:
        """
        Accept a visitor for the visitor pattern.
        This method should be implemented by subclasses.
        """
        cls_name = self.__class__.__name__
        mth = getattr(visitor, f"visit_{cls_name}", None)
        if mth is None:
            raise RuntimeError("Expected visitor has function:  `{}`".format(cls_name))
        return mth(self)


class MapPrimitive(EinsumPrimitive):
    def __init__(
        self,
        lhs: FakeData,
        rhs: FakeData,
        einsum_str: str,
        dims_to_map: Union[str, Sequence[str]],
        op: ComputeOperator,
    ) -> None:
        super().__init__([lhs, rhs], einsum_str)

        # init scripts
        assert len(self.input_scripts) == 2
        self.lhs_scripts, self.rhs_scripts = self.input_scripts
        self.dims_to_map = (
            [dims_to_map] if isinstance(dims_to_map, str) else list(dims_to_map)
        )
        self.lhs = lhs
        self.rhs = rhs
        self.op = op
        self.output = self.run()


class ReducePrimitive(EinsumPrimitive):

    def __init__(
        self,
        x: FakeData,
        einsum_str: str,
        dims_to_reduce: Union[str, Sequence[str]],
        op: ComputeOperator,
    ) -> None:
        super().__init__([x], einsum_str)
        assert len(self.input_scripts) == 1
        self.x_scripts = self.input_scripts[0]  # only one input

        self.dims_to_reduce = (
            [dims_to_reduce]
            if isinstance(dims_to_reduce, str)
            else list(dims_to_reduce)
        )

        # 自己组合出ReduceFu
        self.op = op
        self.output = self.run()


class UnaryPrimitive(EinsumPrimitive):

    def __init__(self, x: FakeData, einsum_str: str, op: ComputeOperator):
        super().__init__(inputs=[x], einsum_str=einsum_str)
        self.x = x
        assert len(self.input_scripts) == 1
        self.x_scripts = self.input_scripts[0]  # only one input
        self.op = op
        self.output = self.run()


class CascadePrimitive(EinsumPrimitive):
    def __init__(
        self,
        inputs: Sequence[FakeData],
        graph,
        einsum_str: str,
    ):
        from Aipiler.graph import EinsumGraph

        super().__init__(list(inputs), einsum_str)
        self.graph: EinsumGraph = graph


class PopulatePrimitive(EinsumPrimitive):

    def __init__(self):
        super().__init__(inputs=[], einsum_str="")
        pass


class EinsumBuilder:
    """
    A builder for creating Einsum primitives.
    This class is used to create Einsum primitives like Map, Reduce, Populate, and Unary.
    """

    @staticmethod
    def map(
        lhs: FakeData,
        rhs: FakeData,
        einsum_str: str,
        dims_to_map: str,
        op: ComputeOperator,
    ) -> FakeData:
        assert lhs.dtype == rhs.dtype
        m = MapPrimitive(lhs, rhs, einsum_str, dims_to_map, op)
        return m.output

    @staticmethod
    def reduce(
        x: FakeData, einsum_str: str, dim_to_reduce: str, op: ComputeOperator
    ) -> FakeData:
        return ReducePrimitive(x, einsum_str, dim_to_reduce, op).output

    @staticmethod
    def unary(x: FakeData, einsum_str: str, op: ComputeOperator) -> FakeData:
        return UnaryPrimitive(x, einsum_str, op).output

    @staticmethod
    def cascade(
        *funcs_and_params: Tuple[Callable, Sequence[Union[str, FakeData]]],
        output_idx: Union[Sequence[int], int] = -1,
    ) -> Union[FakeData, Tuple[FakeData]]:
        """
        cascade is a black box / subgraph in einsum graph
        example (all map prims are element-wise add):
                 (E)
                  |                                  (E)
                |map|      # map3                     |
                /   \                             [cascade] ij,ij->ij
            |map|   |map|  # map1  map2  ==>     /   | |   \  
            /   \   /  \                        /   |   |   \
          (C)   |map|  (D) # map0             (C)  (A) (B)  (D)
                 / \ 
               (A) (B)                             
        
        @einsum
        def test():
            ... # prepare A, B, C, D
            @cascade
            def four_add(A, B, C, D):
                t0 = map(A, B, "ij, ij -> ij", "+")
                t1 = map(C, t0, "ij, ij -> ij", "+")
                t2 = map(t0, D, "ij, ij -> ij", "+")
                t3 = map(t1, t2, "ij, ij -> ij", "+")
                return t3
            E = four_add(A, B, C, D)
            ...
        """
        from Aipiler.graph import EinsumGraph, trace_from, einsum_str_from_graph

        # interpreter
        cascade_inputs: List[FakeData] = []
        graph_outputs: List[FakeData] = []
        intermediate_var = []

        for idx, (fun, params) in enumerate(funcs_and_params):
            # collect fun_args
            fun_args = []
            for param in params:
                if isinstance(param, int):
                    if param >= idx:
                        raise ValueError(
                            f"{idx}-th function in `cascade` uses the return from the {param}-th"
                        )
                    fun_args.append(intermediate_var[param])
                else:
                    # assert param is FakeDate
                    if not isinstance(param, FakeData):
                        raise ValueError(
                            "`cascade` expects `int` or `FakeData` as input, got {}".format(
                                type(param)
                            )
                        )
                    fun_args.append(param)
                    # collect inputs of cascade_inputs
                    if param not in cascade_inputs:
                        cascade_inputs.append(param)
            iv = fun(*fun_args)
            intermediate_var.append(iv)

        if isinstance(output_idx, int):
            output_idx = [output_idx]
        else:
            output_idx = list(output_idx)

        # get output of cascade
        for idx in output_idx:
            if idx >= len(intermediate_var) or idx < 0:
                raise ValueError(
                    "output_idx of `cascade` is expected to be 0 <= output_idx < {}".format(
                        len(intermediate_var)
                    )
                )
            graph_outputs.append(intermediate_var[idx])

        # trace graph
        graph_inputs: List[FakeData] = []
        for inp in cascade_inputs:
            graph_input = copy(inp)
            if isinstance(graph_input, FakeTensor):
                graph_input._trace = None
            graph_inputs.append(graph_input)

        graph = trace_from(graph_outputs, graph_inputs)
        # assert no node in graph is cascade
        for node in graph.nodes:
            if isinstance(node, CascadePrimitive):
                raise NotImplementedError("Unexpected nested `cascade`")
        # create prim
        # TODO: build map between:
        # cascade input  <--> graph input
        # cascade output <--> graph output
        prim = CascadePrimitive(cascade_inputs, graph, einsum_str_from_graph(graph))
        return tuple(prim.output)

    @staticmethod
    def populate() -> FakeData:
        pass
