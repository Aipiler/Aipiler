from abc import ABC, abstractmethod
from executionStrategy import ExecutionStrategy
from typing import Dict, Any, List, Type, Optional, Set, Tuple, Callable
from enum import Enum, auto
from data import Data


class AlgebraicProperty(Enum):
    """代数性质枚举"""

    COMMUTATIVE = auto()  # 交换律: a * b = b * a
    ASSOCIATIVE = auto()  # 结合律: (a * b) * c = a * (b * c)
    DISTRIBUTIVE = auto()  # 分配律: a * (b + c) = a * b + a * c
    IDEMPOTENT = auto()  # 幂等性: a * a = a
    IDENTITY = auto()  # 具有单位元: a * e = a
    INVERTIBLE = auto()  # 可逆性: a * a^(-1) = e
    ABSORBING = auto()  # 吸收律: a * 0 = 0
    NILPOTENT = auto()  # 幂零性: a^n = 0 对某个 n
    MONOTONIC = auto()  # 单调性: a ≤ b => f(a) ≤ f(b)
    DETERMINISTIC = auto()  # 确定性: 相同输入总是产生相同输出
    DIFFERENTIABLE = auto()  # 可微分性: 支持梯度计算


class Port(ABC):
    """定义计算端口的接口"""

    @abstractmethod
    def get_input_schema(self) -> List[Data]:
        """返回输入数据"""
        pass

    @abstractmethod
    def get_output_schema(self) -> List[Data]:
        """返回输出数据"""
        pass

    @abstractmethod
    def verify_inputs(self, inputs: List[Data]) -> bool:
        """验证输入数据的有效性"""
        pass

    @abstractmethod
    def infer_output_shapes(self, inputs: List[Data]) -> List[Data]:
        """根据输入数据形状推断输出数据形状"""
        pass

    @abstractmethod
    def get_algebraic_properties(self) -> Dict[AlgebraicProperty, bool]:
        """返回此计算操作支持的代数性质"""
        pass

    @abstractmethod
    def get_identity_element(self, property_type: AlgebraicProperty) -> Optional[Any]:
        """返回特定代数性质的单位元（如果存在）

        例如，对于加法，单位元是0；对于乘法，单位元是1
        """
        pass

    @abstractmethod
    def is_equivalent_to(self, other: "Port") -> bool:
        """检查此Port是否在数学上等价于另一个Port

        用于确定是否可以合并或替换操作
        """
        pass


class Compute(Port):
    """实现Port接口的计算单元基类"""

    def __init__(self, name: str = "Compute"):
        self.name = name

    def get_input_schema(self) -> Dict[str, Type]:
        """子类应覆盖此方法，提供输入数据的结构定义"""
        return {}

    def get_output_schema(self) -> Dict[str, Type]:
        """子类应覆盖此方法，提供输出数据的结构定义"""
        return {}

    def verify_inputs(self, inputs: Dict[str, Any]) -> bool:
        """验证输入数据的有效性"""
        pass

    def infer_output_shapes(self, input_shapes: Dict[str, Tuple]) -> Dict[str, Tuple]:
        """根据输入形状推断输出形状，默认实现返回空字典"""
        return {}

    def get_algebraic_properties(self) -> Dict[AlgebraicProperty, bool]:
        """返回此计算操作支持的代数性质

        默认情况下，所有性质都是False。子类应根据实际情况覆盖此方法。
        """
        return {prop: False for prop in AlgebraicProperty}

    def get_identity_element(self, property_type: AlgebraicProperty) -> Optional[Any]:
        """返回特定代数性质的单位元

        默认返回None，表示没有单位元。子类应根据实际情况覆盖此方法。
        """
        return None

    def is_equivalent_to(self, other: "Port") -> bool:
        """检查此Port是否在数学上等价于另一个Port

        默认实现只检查是否是同一类。子类应根据实际情况覆盖此方法。
        """
        return isinstance(other, self.__class__)

    def add_strategy(self, strategy):
        """注册进新的执行策略"""
        self.strategies.append(strategy)

    def codeGen(self, input_edges, output_edge):
        """执行最优策略"""
        # 选择最适合当前情况的策略
        # best_strategy = self.select_best_strategy(input_edges)
        pass

    def select_best_strategy(self, input_data):
        """选择最佳执行策略"""
        # 基于输入大小、硬件特性等选择最佳策略

    @abstractmethod
    def _compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """具体的计算逻辑，需要子类实现"""
        pass


class Matmul(Compute):
    """矩阵乘法操作 - 算法层面的定义"""

    """Tensor-level, Tile-level, Vector-level, cpu, gpu等等
        实现算子的注册机制，按照不同分类，注册进去。
    """

    def __init__(self):
        super().__init__()
        self.attributes = {"operation": "matmul"}

    def codeGen(self, a, b, out=None, **params):
        """执行矩阵乘法，可适应不同大小的输入"""
        pass


class Conv2D(Compute):
    """二维卷积操作 - 算法层面的定义"""

    def __init__(self):
        super().__init__()
        self.attributes = {"operation": "conv2d"}

    def codeGen(self, input_tensor, kernel, out=None, **params):
        """执行二维卷积，可适应不同大小的输入"""
        pass


class ElementWise(Compute):
    """逐元素操作 - 算法层面的定义"""

    def __init__(self):
        super().__init__()
        self.attributes = {}


class Reduction(Compute):
    """归约操作 - 算法层面的定义"""

    def __init__(self):
        super().__init__()
        self.attributes = {}


# 通用计算类，用于处理任何类型的操作
class EinsumExpression:
    """通用计算操作，可以表示任何类型的操作"""

    def __init__(self, name):
        self.name = name

    def verify(self, inputs: List[Data]) -> Tuple[bool, Optional[str]]:
        # 通用验证，总是返回成功
        return True, None

    def infer_shape(self, input_shapes: List[Any]) -> Optional[Tuple]:
        # 通用形状推断，默认保持与第一个输入相同
        if not input_shapes or input_shapes[0] is None:
            return None
        return input_shapes[0]
