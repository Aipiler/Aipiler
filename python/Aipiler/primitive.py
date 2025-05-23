from Aipiler.tensor import Tensor
from Aipiler.basic_operator import BasicOperator
from typing import List, Union, Sequence



 
class EinsumPrimitive:
    def __init__(self, inputs: List[Tensor], einsum_str: str) -> None:
        self.inputs = inputs
        self.einsum_str = einsum_str
        self.output: Tensor = None
        
    def run(self):
        raise NotImplementedError()


class Map(EinsumPrimitive):
    def __init__(self, lhs: Tensor, rhs: Tensor, einsum_str: str, ranks_to_map: Union[str, Sequence[str]], op: BasicOperator) -> None:
        super().__init__([lhs, rhs], einsum_str)
        self.einsum_str = einsum_str
        self.ranks_to_map = [ranks_to_map] if isinstance(ranks_to_map, str) else list(ranks_to_map)
        self.op = op
        self.output = self.run()
    
    def run(self):
        """
        check inputs and einsum, generate symbolic outputs 
        """
        # TODO: get dim obj by einsum str
        return Tensor([], self)


class Reduce(EinsumPrimitive):
    def __init__(self, x: Tensor, einsum_str: str, rank_to_reduce: str, op: BasicOperator) -> None:
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
    pass


def map(lhs: Tensor, rhs: Tensor, einsum_str: str, rank_to_map: str, op: BasicOperator):
    m = Map(lhs, rhs, einsum_str, rank_to_map, op)
    return m.output


def reduce(x: Tensor, einsum_str: str, rank_to_reduce: str, op: BasicOperator):
    return Reduce(x, einsum_str, rank_to_reduce, op).output


def populate():
    pass


def unary():
    pass