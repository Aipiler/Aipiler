import torch
from .Eoperator import Operator, EinsumOperator, NamedOperators
from typing import Callable, Optional, Tuple, List, Dict, Any, Union
from .einsum.einsumExpression import (
    EinsumExpression,
    MapEquation,
    ReduceEquation,
    RankMap,
)
from .einsum.tensor import Tensor
from .einsum.rankVariable import RankVariable
from .einsum.term import AffineTerm, VarTerm
from .einsum.rankExpression import (
    AffineRankExpression,
    SimpleRankExpression,
    NonAffineRankRank,
)
from .einsum.range import Range, CompoundRange
from .einsum.constraint import (
    Constraint,
    StaticConstraint,
    DynamicConstraint,
    ComparisonOperator,
)
from .einsum.operators.compute import ComputeOperator
from .einsum.operators.unary import UnaryOperator
from .einsum.operators.coordinate import CoordinateOperator
from .einsum.operators.merge import MergeOperator


# 管理映射的类
class PyTorchOpMapper:
    def __init__(self):
        self._mapping = {}

    def register(self, aten_op_name: str):
        def decorator(handler_func):
            print(f"Registering handler for: {aten_op_name}")
            self._mapping[aten_op_name] = handler_func
            return handler_func

        return decorator

    def get_handler(self, aten_op_name: str) -> Callable:
        handler = self._mapping.get(aten_op_name)
        if handler is None:
            raise NotImplementedError(
                f"No handler registered for ATen operator: {aten_op_name}"
            )
        return handler


einsum_mapper = PyTorchOpMapper()


# --- 注册具体的处理函数 ---


@einsum_mapper.register("aten.max_pool2d.default")  # 注册 Add 算子的处理器
def handle_add(node: torch.fx.Node) -> Operator:
    # 从 node (例如 torch.fx.Node) 提取信息
    # 注意：实际中需要准确解析 args 和 kwargs 来确定输入和属性
    print(f"Handling aten::add.Tensor from node: {node.name}")
    # TODO: 这里通过 node.args 和 node.kwargs 获取参数

    kernel_size = node.args[1][0]
    stride = node.args[2][0]
    tensor_input = Tensor("A", (10, 10))
    tensor_output = Tensor("B", (5, 5))
    rankMap = RankMap()
    varM = RankVariable("m")
    varN = RankVariable("n")
    varK = RankVariable("k")
    varK.add_constraint(
        StaticConstraint(varK, ComparisonOperator.LESS_THAN, kernel_size)
    )
    rankMap.add_mapping(
        tensor_input.get_i_rank(0),
        AffineRankExpression(
            affineTerm=AffineTerm(kernel_size, VarTerm(varM, stride), VarTerm(varK))
        ),
    )
    rankMap.add_mapping(
        tensor_input.get_i_rank(1),
        AffineRankExpression(
            affineTerm=AffineTerm(kernel_size, VarTerm(varN, stride), VarTerm(varK))
        ),
    )
    rankMap.add_mapping(
        tensor_output.get_i_rank(0),
        AffineRankExpression(affineTerm=AffineTerm(kernel_size, VarTerm(varM, stride))),
    )
    rankMap.add_mapping(
        tensor_output.get_i_rank(1),
        AffineRankExpression(affineTerm=AffineTerm(kernel_size, VarTerm(varM, stride))),
    )
    maxPooling = ReduceEquation(
        tensor_output, tensor_input, rankMap, [varK], ComputeOperator.MAX
    )
    return EinsumOperator(einsum_expr=maxPooling)


@einsum_mapper.register("aten.relu.default")  # 注册 MatMul 算子的处理器
def handle_matmul(pytorch_node: torch.fx.Node) -> Operator:
    print(f"Handling aten::matmul from node: {pytorch_node.name}")
    # 可能需要检查 pytorch_node.args 来确定是否有转置等情况
    # 调用你的 MatMul 工厂函数
    return MatMul()  # 返回配置好的自定义 MatMul 算子实例


@einsum_mapper.register("aten.conv2d.default")  # 注册 MatMul 算子的处理器
def handle_matmul(pytorch_node: torch.fx.Node) -> Operator:
    print(f"Handling aten::matmul from node: {pytorch_node.name}")
    # 可能需要检查 pytorch_node.args 来确定是否有转置等情况
    # 调用你的 MatMul 工厂函数
    return MatMul()  # 返回配置好的自定义 MatMul 算子实例
