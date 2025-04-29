import torch
from .Eoperator import Operator, EinsumOperator, NamedOperators
from typing import Callable, Optional, Tuple, List, Dict, Any, Union


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


@einsum_mapper.register("aten::add.Tensor")  # 注册 Add 算子的处理器
def handle_add(pytorch_node: torch.fx.Node) -> Operator:
    # 从 pytorch_node (例如 torch.fx.Node) 提取信息
    # 注意：实际中需要准确解析 args 和 kwargs 来确定输入和属性
    print(f"Handling aten::add.Tensor from node: {pytorch_node.name}")

    # TODO: 创建一个EinsumExpression
    einsum_expr = None  # TODO: 这里需要根据实际情况创建一个EinsumExpression实例
    return EinsumOperator(einsum_expr=einsum_expr)


@einsum_mapper.register("aten::matmul")  # 注册 MatMul 算子的处理器
def handle_matmul(pytorch_node: torch.fx.Node) -> Operator:
    print(f"Handling aten::matmul from node: {pytorch_node.name}")
    # 可能需要检查 pytorch_node.args 来确定是否有转置等情况
    # 调用你的 MatMul 工厂函数
    return MatMul()  # 返回配置好的自定义 MatMul 算子实例
