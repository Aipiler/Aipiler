from Aipiler.tensor import Tensor
from Aipiler.basic_operator import BaseOperator
from Aipiler.dim import Dim, AffineDimExpr
from typing import List, Union, Sequence, Dict, Any
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from Aipiler.utils import parse_einsum_str

if TYPE_CHECKING:
    from Aipiler.visitor import Visitor


class EinsumPrimitive(ABC):
    def __init__(self, inputs: List[Tensor], einsum_str: str) -> None:
        self.inputs = inputs
        self.einsum_str = einsum_str
        self.output: Tensor = None

    def run(self):
        """
        check inputs and einsum, generate symbolic outputs
        """
        from .utils import parse_einsum_str

        input_scripts, output_scripts = parse_einsum_str(self.einsum_str)

        # get map of `str -> dim obj`

        # create output
        assert len(self.inputs) == len(input_scripts)
        tensor_shape: List[Dim] = []
        for output_script in output_scripts:
            # affine expr of dim object
            affine_exprs = []
            for input_, tensor_input in zip(input_scripts, self.inputs):
                for idx, input_script in enumerate(input_):
                    # if script appear in both output_script and input_script
                    if output_script == input_script:
                        # construct AffineDimExpr
                        input_dim = tensor_input.symbolic_shape[idx]
                        if input_dim.affine_exprs:
                            affine_exprs += input_dim.affine_exprs
                            affine_exprs = list(set(affine_exprs))
                        else:
                            affine_expr = AffineDimExpr(input_dim)
                            affine_exprs.append(affine_expr)
            # construct dim obj
            assert affine_exprs
            tensor_shape.append(Dim(affine_exprs))
            dtype = self.inputs[0].dtype
        return Tensor(tensor_shape, dtype, self)

    @abstractmethod
    def accept(self, visitor: Visitor) -> None:
        """
        Accept a visitor for the visitor pattern.
        This method should be implemented by subclasses.
        """
        pass


class MapPrimitive(EinsumPrimitive):

    def __init__(
        self,
        lhs: Tensor,
        rhs: Tensor,
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
        self.ranks_to_map = (
            [dims_to_map] if isinstance(dims_to_map, str) else list(dims_to_map)
        )
        self.op = op
        self.output = self.run()

    def accept(self, visitor: Visitor) -> Any:
        return visitor.visit_map(self)


class ReducePrimitive(EinsumPrimitive):

    def __init__(
        self,
        x: Tensor,
        einsum_str: str,
        dims_to_reduce: Union[str, Sequence[str]],
        op: BaseOperator,
    ) -> None:
        super().__init__([x], einsum_str)
        self.x_scripts = parse_einsum_str(einsum_str)[0][0]  # only one input
        self.output_scripts = parse_einsum_str(einsum_str)[1]
        self.iteration_scripts = set(self.x_scripts + self.output_scripts)
        self.ranks_to_reduce = (
            [dims_to_reduce]
            if isinstance(dims_to_reduce, str)
            else list(dims_to_reduce)
        )
        self.op = op
        self.output = self.run()

    def accept(self, visitor: Visitor) -> Any:
        return visitor.visit_reduce(self)


# TODO
class PopulatePrimitive(EinsumPrimitive):

    def __init__(self):
        super().__init__(inputs=[], einsum_str="")
        pass

    def accept(self, visitor: Visitor) -> Any:
        return visitor.visit_populate(self)


class UnaryPrimitive(EinsumPrimitive):

    def __init__(self, x: Tensor, op: BaseOperator):
        super().__init__(inputs=[x], einsum_str="")
        self.op = op
        self.output = self.run()

    def accept(self, visitor: Visitor) -> Any:
        return visitor.visit_unary(self)


class EinsumBuilder:
    """
    A builder for creating Einsum primitives.
    This class is used to create Einsum primitives like Map, Reduce, Populate, and Unary.
    """

    @staticmethod
    def map(
        lhs: Tensor, rhs: Tensor, einsum_str: str, dims_to_map: str, op: BaseOperator
    ) -> Tensor:
        m = MapPrimitive(lhs, rhs, einsum_str, dims_to_map, op)
        return m.output

    @staticmethod
    def reduce(
        x: Tensor, einsum_str: str, dim_to_reduce: str, op: BaseOperator
    ) -> Tensor:
        return ReducePrimitive(x, einsum_str, dim_to_reduce, op).output

    @staticmethod
    def populate() -> Tensor:
        pass

    @staticmethod
    def unary(x: Tensor, op: BaseOperator) -> Tensor:
        return UnaryPrimitive(x, op).output
