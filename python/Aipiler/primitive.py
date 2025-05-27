from Aipiler.tensor import Tensor
from Aipiler.basic_operator import BaseOperator
from typing import List, Union, Sequence


class EinsumPrimitive:
    def __init__(self, inputs: List[Tensor], einsum_str: str) -> None:
        self.inputs = inputs
        self.einsum_str = einsum_str
        self.output: Tensor = None

    def run(self):
        raise NotImplementedError()


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

    def run(self):
        """
        check inputs and einsum, generate symbolic outputs
        """
        # TODO: get dim obj by einsum str
        return Tensor([], self)


class Reduce(EinsumPrimitive):

    def __init__(
        self, x: Tensor, einsum_str: str, rank_to_reduce: str, op: BaseOperator
    ) -> None:
        super().__init__([x], einsum_str)
        self.reduce_rank = rank_to_reduce
        self.op = op
        self.output = self.run()

    def run(self):
        """
        check inputs and einsum, generate symbolic outputs
        """
        # TODO: get dim obj by einsum str
        return Tensor([], self)


class Populate(EinsumPrimitive):
    pass


class Unary(EinsumPrimitive):

    def __init__(self, x: Tensor, op: BaseOperator):
        super().__init__(inputs=[x], einsum_str="")
        self.op = op
        self.output = self.run()

    def run(self):
        """
        check inputs and einsum, generate symbolic outputs
        """
        return Tensor([], self)


def map(lhs: Tensor, rhs: Tensor, einsum_str: str, ranks_to_map: str, op: BaseOperator):
    m = Map(lhs, rhs, einsum_str, ranks_to_map, op)
    return m.output


def reduce(x: Tensor, einsum_str: str, rank_to_reduce: str, op: BaseOperator):
    return Reduce(x, einsum_str, rank_to_reduce, op).output


def populate():
    pass


def unary(x: Tensor, op: BaseOperator):
    return Unary(x, op).output
