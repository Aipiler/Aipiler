from Aipiler.tensor import Tensor
from Aipiler.basic_operator import BaseOperator
from Aipiler.dim import Dim, AffineDimExpr
from typing import List, Union, Sequence, Dict


class EinsumPrimitive:
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


class Map(EinsumPrimitive):

    def __init__(
        self,
        lhs: Tensor,
        rhs: Tensor,
        einsum_str: str,
        ranks_to_map: Union[str, Sequence[str]],
        op: BaseOperator,
    ) -> None:
        super().__init__([lhs, rhs], einsum_str)
        self.einsum_str = einsum_str
        self.ranks_to_map = (
            [ranks_to_map] if isinstance(ranks_to_map, str) else list(ranks_to_map)
        )
        self.op = op
        self.output = self.run()


class Reduce(EinsumPrimitive):

    def __init__(
        self,
        x: Tensor,
        einsum_str: str,
        ranks_to_reduce: Union[str, Sequence[str]],
        op: BaseOperator,
    ) -> None:
        super().__init__([x], einsum_str)
        self.ranks_to_reduce = (
            [ranks_to_reduce]
            if isinstance(ranks_to_reduce, str)
            else list(ranks_to_reduce)
        )
        self.op = op
        self.output = self.run()


class Populate(EinsumPrimitive):
    pass


class Unary(EinsumPrimitive):
    def __init__(self, x: Tensor, op: BaseOperator):
        super().__init__(inputs=[x], einsum_str="")
        self.op = op
        self.output = self.run()


def map(
    lhs: Tensor,
    rhs: Tensor,
    einsum_str: str,
    ranks_to_map: Union[str, Sequence[str]],
    op: BaseOperator,
):
    return Map(lhs, rhs, einsum_str, ranks_to_map, op).output


def reduce(x: Tensor, einsum_str: str, rank_to_reduce: str, op: BaseOperator):
    return Reduce(x, einsum_str, rank_to_reduce, op).output


def populate():
    pass


def unary(x: Tensor, op: BaseOperator):
    return Unary(x, op).output
