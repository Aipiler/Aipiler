from .einsumExpression import EinsumExpression
from .tensor import DataSpace, Rank, RankSet
from .rankVariable import RankVariable, RankVariableSet
from .rankExpression import RankMap, RankExpression, AffineRankExpression
from .einsumExpression import (
    MapEquation,
    ReduceEquation,
    PopulateEquation,
    UnaryEquation,
)
from .term import ConstTerm, VarTerm, AffineTerm
from .operators.compute import ComputeOperator
from .operators.coordinate import CoordinateOperator
from .operators.unary import UnaryOperator


class Context:
    """管理所有配置信息，包括注册的computeOp，unaryOp和emptyType等等。"""

    def __init__(self):
        self.einsum_list = []
        self.data_list = []
        self.input_data_space_list = []
        self.output_data_space_list = []

    def register_input_data_space(self, input_data_spaces: list[DataSpace]):
        self.input_data_space = input_data_spaces

    def register_output_data_space(self, output_data_spaces: list[DataSpace]):
        self.output_data_space_list = output_data_spaces

    def add_einsum(self, einsum: EinsumExpression):
        self.einsum_list.append(einsum)

    def add_data(self, data: DataSpace):
        self.data_list.appned(data)


class Builder:
    def __init__(self, context: Context):
        self.context = context

    def get_rank_set(self, shape: tuple[int]) -> RankSet:
        ranks = []
        for s in shape:
            ranks.append(Rank(s))

        rankset = RankSet(ranks=ranks)
        return rankset

    def gete_rank_variable(self) -> RankVariable:
        return RankVariable()

    def get_rank_varaible_set(
        self, *rank_variable: tuple[RankVariable, ...]
    ) -> RankVariableSet:
        rkvs = RankVariableSet()
        for rv in rank_variable:
            rkvs.add_rankVariable()
        return rkvs

    def get_const_term(self, value: int) -> ConstTerm:
        return ConstTerm(value)

    def get_var_term(self, rankVariable: RankVariable, coefficient: int = 1) -> VarTerm:
        return VarTerm(rankVariable, coefficient)

    def get_affine_rank_expression(
        self, const_term: ConstTerm, *var_terms: tuple[VarTerm, ...]
    ) -> RankExpression:
        return AffineRankExpression(
            AffineTerm(constTerm=const_term, varTerms=list(var_terms))
        )

    def get_rank_map(self) -> RankMap:
        return RankMap()

    def create_map_equation(
        self,
        output_rank_set: RankSet,
        first_rank_set: RankSet,
        second_rank_set: RankSet,
        rankVariable_set: RankVariableSet,
        rankMap: RankMap,
        target_ranks: list[RankVariable],
        computeOp: ComputeOperator,
    ) -> DataSpace:
        map_einsum = MapEquation(
            output_rank_set,
            first_rank_set,
            second_rank_set,
            rankVariable_set,
            rankMap,
            target_ranks,
            computeOp,
        )
        new_data_space = DataSpace(map_einsum, output_rank_set)
        self.context.add_einsum(map_einsum)
        self.context.add_data(new_data_space)
        return new_data_space

    def create_reduce_equation(
        self,
        output_rank_set: RankSet,
        input_rank_set: RankSet,
        rankVariable_set: RankVariableSet,
        rankMap: RankMap,
        target_ranks: list[RankVariable],
        computeOp: ComputeOperator,
    ) -> DataSpace:
        reduce_einsum = ReduceEquation(
            output_rank_set,
            input_rank_set,
            rankVariable_set,
            rankMap,
            target_ranks,
            computeOp,
        )
        new_data_space = DataSpace(reduce_einsum, output_rank_set)
        self.context.add_einsum(reduce_einsum)
        self.context.add_data(new_data_space)
        return new_data_space

    def create_Populate_equation(
        self,
        output_rank_set: RankSet,
        input_rank_set: RankSet,
        rankVariable_set: RankVariableSet,
        rankMap: RankMap,
        target_ranks: list[RankVariable],
        computeOp: ComputeOperator,
        coordinateOp: CoordinateOperator,
    ) -> DataSpace:
        # 设置创建出来的dataSpace的def einsum为本map
        pass

    def create_unary_equation(
        self,
        output_rank_set: RankSet,
        input_rank_set: RankSet,
        rankVariable_set: RankVariableSet,
        rankMap: RankMap,
        unaryOp: UnaryOperator,
    ) -> DataSpace:
        unary_einsum = UnaryEquation(
            output_rank_set, input_rank_set, rankVariable_set, rankMap, unaryOp
        )
        new_data_space = DataSpace(unary_einsum, output_rank_set)
        self.context.add_einsum(unary_einsum)
        self.context.add_data(new_data_space)
        return new_data_space


def attention():
    pass


def RMSNorm_Matmul():
    pass


if __name__ == "__main__":
    attention()
