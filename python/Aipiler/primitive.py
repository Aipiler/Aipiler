from Aipiler.tensor import FakeTensor, FakeData, FakeScalar
from Aipiler.basic_operator import ComputeOperator
from Aipiler.dim import Dim, dims
from typing import List, Union, Sequence, Dict, Any
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from Aipiler.utils import parse_einsum_str


class EinsumPrimitive(ABC):
    def __init__(self, inputs: List[FakeData], einsum_str: str) -> None:
        self.inputs = inputs
        self.einsum_str = einsum_str
        self.output: FakeData = None
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

        # get map of `str -> dim obj`

        # create output
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
    def populate() -> FakeData:
        pass
