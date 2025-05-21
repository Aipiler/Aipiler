from .rankExpression import (
    RankExpression,
    SimpleRankExpression,
    AffineRankExpression,
    NonAffineRankRank,
)
from .rankVariable import RankVariable, RankVariableSet
from .term import AffineTerm, VarTerm, ConstTerm
from .operators.compute import ComputeOperator
from .operators.unary import UnaryOperator
from .operators.coordinate import CoordinateOperator
from .operators.merge import MergeOperator
from .tensor import RankSet, DataSpace
from .range import Range, CompoundRange
from .rankExpression import RankMap
from .constraint import (
    Constraint,
    StaticConstraint,
    DynamicConstraint,
    ComparisonOperator,
)
from typing import List, Optional, Tuple, Dict, Any, Set, Callable, Union, TypeVar
from abc import ABC, abstractmethod


T = TypeVar("T", bound="EinsumExpression")


class EinsumExpression(ABC):
    """Base class for all Einsum expressions."""

    def __init__(
        self,
        outputs_rank_set: List[RankSet],
        inputs_rank_set: List[RankSet],
    ):
        self.output_rank_set = outputs_rank_set
        self.inputs_rank_set = inputs_rank_set


class EinsumEquation(EinsumExpression):
    """Represents an EDGE Einsum expression."""

    def __init__(
        self,
        output_rank_set: RankSet,
        inputs_rank_set: List[RankSet],
        rankVariable_set: RankVariableSet,
        rankMap: RankMap,
        target_ranks: List[RankVariable],
    ):
        # Extract the output tensor and input tensors from the mappings
        super().__init__(
            outputs_rank_set=[output_rank_set],
            inputs_rank_set=inputs_rank_set,
        )
        self.rankMap = rankMap
        self.rankVariable_set = rankVariable_set
        self.target_ranks = target_ranks

    def gen_iteration_domain(self):
        """Generate the iteration domain for the Einsum equation.

        Returns:
            An object representing the iteration domain.
        """
        # TODO: 实现迭代域生成逻辑
        raise NotImplementedError("This method needs to be implemented")


class MapEquation(EinsumEquation):
    """Represents a Map equation in the Einsum expression."""

    def __init__(
        self,
        output_rank_set: RankSet,
        first_rank_set: RankSet,
        second_rank_set: RankSet,
        rankVariable_set: RankVariableSet,
        rankMap: RankMap,
        target_ranks: List[RankVariable],
        computeOp: ComputeOperator,
    ):
        super().__init__(
            output_rank_set=output_rank_set,
            inputs_rank_set=[first_rank_set, second_rank_set],
            rankVariable_set=rankVariable_set,
            rankMap=rankMap,
            target_ranks=target_ranks,
        )
        self.first_rank_set = first_rank_set
        self.second_rank_set = second_rank_set
        self.computeOp = computeOp

    def __repr__(self) -> str:
        pass


class ReduceEquation(EinsumEquation):
    """Represents a Reduce equation in the Einsum expression."""

    def __init__(
        self,
        output_rank_set: RankSet,
        input_rank_set: RankSet,
        rankVariable_set: RankVariableSet,
        rankMap: RankMap,
        target_ranks: List[RankVariable],
        computeOp: ComputeOperator,
    ):
        super().__init__(
            output_rank_set=output_rank_set,
            input_rank_set=[input_rank_set],
            rankVariable_set=rankVariable_set,
            rankMap=rankMap,
            target_ranks=target_ranks,
        )
        self.input_rank_set = input_rank_set
        self.computeOp = computeOp


class PopulateEquation(EinsumEquation):
    """Represents a Populate equation in the Einsum expression."""

    def __init__(
        self,
        output_rank_set: RankSet,
        input_rank_set: RankSet,
        rankVariable_set: RankVariableSet,
        rankMap: RankMap,
        target_ranks: List[RankVariable],
        computeOp: ComputeOperator,
        coordinateOp: CoordinateOperator,
    ):
        super().__init__(
            output_rank_set=output_rank_set,
            input_rank_set=[input_rank_set],
            rankVariable_set=rankVariable_set,
            rankMap=rankMap,
            target_ranks=target_ranks,
        )
        self.input_rank_set = input_rank_set
        self.computeOp = computeOp
        self.coordinateOp = coordinateOp


class UnaryEquation(EinsumEquation):
    """Represents a Unary equation in the Einsum expression."""

    def __init__(
        self,
        output_rank_set: RankSet,
        input_rank_set: RankSet,
        rankVariable_set: RankVariableSet,
        rankMap: RankMap,
        unaryOp: UnaryOperator,
    ):
        super().__init__(
            output_rank_set=output_rank_set,
            input_rank_set=[input_rank_set],
            rankVariable_set=rankVariable_set,
            rankMap=rankMap,
            target_ranks=rankVariable_set.get_rankVariables(),
        )
        self.input_rank_set = input_rank_set
        self.unaryOp = unaryOp


class EinsumCascade(EinsumExpression):
    """Represents a cascade of Einsum equations."""

    def __init__(
        self,
        outputs_rank_set: List[RankSet],
        inputs_rank_set: List[RankSet],
        equations: List[EinsumEquation],
    ):
        super().__init__(
            outputs_rank_set=outputs_rank_set,
            inputs_rank_set=inputs_rank_set,
        )
        self.equations = equations


class EinsumIteration(EinsumCascade):
    """Represents an iteration in the Einsum expression."""

    def __init__(
        self,
        outputs_rank_set: List[RankSet],
        inputs_rank_set: List[RankSet],
        equations: List[EinsumEquation],
        generative_rank: RankVariable,
    ):
        super().__init__(
            outputs_rank_set=outputs_rank_set,
            inputs_rank_set=inputs_rank_set,
            equations=equations,
        )
        self.generative_rank = generative_rank


def maxPooling2D():
    pooling_size = 2
    stride = 2
    tensor_input = DataSpace()
    input_rank_set = tensor_input.gen_rank_set_from_shape(shape=(10, 10))
    tensor_output = DataSpace()
    output_rank_set = tensor_output.gen_rank_set_from_shape(shape=(5, 5))

    varM = RankVariable("m")
    varN = RankVariable("n")
    varK = RankVariable("k")
    varK.add_constraint(
        StaticConstraint(varK, ComparisonOperator.LESS_THAN, pooling_size)
    )
    rankVariable_set = RankVariableSet([varM, varN, varK])
    rankMap = RankMap()
    rankMap.add_mapping(
        input_rank_set[0],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(varM, stride), VarTerm(varK))
        ),
    )
    rankMap.add_mapping(
        input_rank_set[1],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(varN, stride), VarTerm(varK))
        ),
    )
    rankMap.add_mapping(
        output_rank_set[0],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(varM, stride))
        ),
    )
    rankMap.add_mapping(
        output_rank_set[1],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(varM, stride))
        ),
    )
    maxPooling = ReduceEquation(
        output_rank_set,
        input_rank_set,
        rankVariable_set,
        rankMap,
        [varK],
        ComputeOperator.MAX,
    )
    print(maxPooling)


def matmul():
    # map
    tensorA = DataSpace()
    rank_set_A = tensorA.gen_rank_set_from_shape(shape=(3, 4))
    tensorB = DataSpace()
    rank_set_B = tensorB.gen_rank_set_from_shape(shape=(4, 5))

    tensorTmp = DataSpace()
    rank_set_tmp = tensorTmp.init_from_other_rank_set(
        RankSet([rank_set_A[0], rank_set_A[1], rank_set_B[1]])
    )

    rank_var_set_map = RankVariableSet(num=3)
    rankMap1 = RankMap()
    rankMap1.add_mapping(
        rank_set_A[0],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(rank_var_set_map[0]))
        ),
    )
    rankMap1.add_mapping(
        rank_set_A[1],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(rank_var_set_map[1]))
        ),
    )
    rankMap1.add_mapping(
        rank_set_B[0],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(rank_var_set_map[1]))
        ),
    )
    rankMap1.add_mapping(
        rank_set_B[1],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(rank_var_set_map[2]))
        ),
    )
    rankMap1.add_mapping(
        rank_set_tmp[0],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(rank_var_set_map[0]))
        ),
    )
    rankMap1.add_mapping(
        rank_set_tmp[1],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(rank_var_set_map[1]))
        ),
    )
    rankMap1.add_mapping(
        rank_set_tmp[2],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(rank_var_set_map[2]))
        ),
    )
    einsum1 = MapEquation(
        rank_set_tmp,
        rank_set_A,
        rank_set_B,
        rank_var_set_map,
        rankMap1,
        [rank_var_set_map[1]],
        ComputeOperator.MUL,
    )

    # reduce
    tensorC = DataSpace()
    rank_set_C = tensorC.init_from_other_rank_set(
        RankSet([rank_set_tmp[0], rank_set_tmp[1]])
    )
    rank_var_set_C = RankVariableSet(num=3)
    rankMap2 = RankMap()
    rankMap2.add_mapping(
        rank_set_tmp[0],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(rank_var_set_C[0]))
        ),
    )
    rankMap2.add_mapping(
        rank_set_tmp[1],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(rank_var_set_C[1]))
        ),
    )
    rankMap2.add_mapping(
        rank_set_tmp[2],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(rank_var_set_C[2]))
        ),
    )
    rankMap2.add_mapping(
        rank_set_C[0],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(rank_var_set_C[0]))
        ),
    )
    rankMap2.add_mapping(
        rank_set_C[1],
        AffineRankExpression(
            affineTerm=AffineTerm(ConstTerm(0), VarTerm(rank_var_set_C[2]))
        ),
    )
    einsum2 = ReduceEquation(
        rank_set_C,
        rank_set_tmp,
        rank_var_set_C,
        rankMap2,
        [rank_var_set_C[1]],
        ComputeOperator.ADD,
    )


# def matmul_einsum():

#     varM = RankVariable("m")
#     varN = RankVariable("n")
#     varK = RankVariable("k")

#     expr_M = AffineRankExpression(affineTerm=AffineTerm(ConstTerm(0), VarTerm(varM)))
#     expr_K = AffineRankExpression(affineTerm=AffineTerm(ConstTerm(0), VarTerm(varK)))
#     expr_N = AffineRankExpression(affineTerm=AffineTerm(ConstTerm(0), VarTerm(varN)))

#     input1_expr_list = [expr_M, expr_K]
#     input2_expr_list = [expr_K, expr_N]
#     output_expr_list = [expr_M, expr_K, expr_N]

#     einsum1 = MapEquation(
#         output_expr_list,
#         input1_expr_list,
#         input2_expr_list,
#         [varK],
#         ComputeOperator.MUL,
#     )

#     varM = RankVariable("m")
#     varN = RankVariable("n")
#     varK = RankVariable("k")

#     expr_M = AffineRankExpression(affineTerm=AffineTerm(ConstTerm(0), VarTerm(varM)))
#     expr_K = AffineRankExpression(affineTerm=AffineTerm(ConstTerm(0), VarTerm(varK)))
#     expr_N = AffineRankExpression(affineTerm=AffineTerm(ConstTerm(0), VarTerm(varN)))

#     input_expr_list = [expr_M, expr_K, expr_N]
#     output_expr_list = [expr_M, expr_N]

#     einsum2 = ReduceEquation(
#         output_expr_list, input_expr_list, [varK], ComputeOperator.ADD
#     )


def conv2d():
    pass


def relu():
    pass


def softmax():
    pass


def RMSNorm():
    pass


if __name__ == "__main__":
    # Example usage
    maxPooling2D()

#
