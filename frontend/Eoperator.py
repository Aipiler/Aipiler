import abc
from frontend.einsum.einsumExpression import EinsumExpression


# 基础 Operator 接口 (或抽象基类)
class Operator(abc.ABC):
    def __init__(self, **kwargs):
        # 可以存储通用属性，如输出数据类型提示等
        self.attributes = kwargs

    @abc.abstractmethod
    def compute(self, *inputs):
        """执行计算的核心方法"""
        pass

    # 可以添加其他通用方法，如 infer_shape, backward 等
    # def infer_shape(self, input_shapes): ...
    # def backward(self, grad_outputs): ...


# 1. Einsum 算子
class EinsumOperator(Operator):

    def __init__(self, einsum_expr: EinsumExpression, **kwargs):
        super().__init__(**kwargs)

        self.einsum_expr = einsum_expr
        # Einsum 特有的属性可以在这里处理或校验

    # 实际运行时调用
    def compute(self, *inputs):
        # 调用后端的 einsum 实现 (例如 NumPy, PyTorch, TensorFlow, JAX)
        # 需要决定后端是全局配置、传递进来还是 Operator 自己管理
        pass


# 2. 具名算子
class NamedOperator(Operator):
    def __init__(self, op_name: str, **kwargs):
        super().__init__(**kwargs)
        self.op_name = op_name
        # kwargs 里可以包含算子的特定属性，如 stride, padding for Conv
        # 可以在这里校验属性

    def compute(self, *inputs):
        # 调用算子注册表/分发器来查找并执行对应的核函数
        pass
