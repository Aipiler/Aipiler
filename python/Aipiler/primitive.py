from Aipiler.tensor import FakeTensor
from Aipiler.basic_operator import BaseOperator
from Aipiler.dim import Dim
from typing import List, Union, Sequence, Dict, Any
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from Aipiler.utils import parse_einsum_str


class EinsumPrimitive(ABC):
    def __init__(self, inputs: List[FakeTensor], einsum_str: str) -> None:
        self.inputs = inputs
        self.einsum_str = einsum_str
        self.output: FakeTensor = None
        self.input_scripts, self.output_scripts = parse_einsum_str(self.einsum_str)

    def run(self):
        """
        check inputs and einsum, generate symbolic outputs
        """

        # get map of `str -> dim obj`

        # create output
        assert len(self.inputs) == len(self.input_scripts)
        fake_tensor_shape: List[Dim] = []
        for output_script in self.output_scripts:
            # construct dim obj
            fake_tensor_shape.append(Dim())
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
        lhs: FakeTensor,
        rhs: FakeTensor,
        einsum_str: str,
        dims_to_map: Union[str, Sequence[str]],
        op: BaseOperator,
    ) -> None:
        super().__init__([lhs, rhs], einsum_str)

        self.lhs_scripts, self.rhs_scripts = parse_einsum_str(einsum_str)[0]
        self.output_scripts = parse_einsum_str(einsum_str)[1]
        self.iteration_scripts = set(
            self.lhs_scripts + self.rhs_scripts + self.output_scripts
        )
        self.dims_to_map = (
            [dims_to_map] if isinstance(dims_to_map, str) else list(dims_to_map)
        )
        self.op = op
        self.output = self.run()


class ReducePrimitive(EinsumPrimitive):

    def __init__(
        self,
        x: FakeTensor,
        einsum_str: str,
        dims_to_reduce: Union[str, Sequence[str]],
        op: BaseOperator,
    ) -> None:
        super().__init__([x], einsum_str)
        self.x_scripts = parse_einsum_str(einsum_str)[0][0]  # only one input
        self.output_scripts = parse_einsum_str(einsum_str)[1]
        self.iteration_scripts = set(self.x_scripts + self.output_scripts)
        self.dims_to_reduce = (
            [dims_to_reduce]
            if isinstance(dims_to_reduce, str)
            else list(dims_to_reduce)
        )
        self.op = op
        self.output = self.run()


# TODO
class PopulatePrimitive(EinsumPrimitive):

    def __init__(self):
        super().__init__(inputs=[], einsum_str="")
        pass


class UnaryPrimitive(EinsumPrimitive):

    def __init__(self, x: FakeTensor, op: BaseOperator):
        super().__init__(inputs=[x], einsum_str="")
        self.op = op
        self.output = self.run()


class EinsumBuilder:
    """
    A builder for creating Einsum primitives.
    This class is used to create Einsum primitives like Map, Reduce, Populate, and Unary.
    """

    @staticmethod
    def map(
        lhs: FakeTensor,
        rhs: FakeTensor,
        einsum_str: str,
        dims_to_map: str,
        op: BaseOperator,
    ) -> FakeTensor:
        m = MapPrimitive(lhs, rhs, einsum_str, dims_to_map, op)
        return m.output

    @staticmethod
    def reduce(
        x: FakeTensor, einsum_str: str, dim_to_reduce: str, op: BaseOperator
    ) -> FakeTensor:
        return ReducePrimitive(x, einsum_str, dim_to_reduce, op).output

    @staticmethod
    def populate() -> FakeTensor:
        pass

    @staticmethod
    def unary(x: FakeTensor, op: BaseOperator) -> FakeTensor:
        return UnaryPrimitive(x, op).output
