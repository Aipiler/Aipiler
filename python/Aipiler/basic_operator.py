import operator
from typing import Callable, Any, Dict, List, Optional, Set
from abc import ABC, abstractmethod
from functools import wraps
import inspect


class OperatorRegistry:
    """操作符注册中心"""

    def __init__(self):
        self._operators: Dict[str, "BaseOperator"] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, name: str, category: str = "general", aliases: List[str] = None):
        """注册操作符的装饰器"""

        def decorator(operator_class_or_func):
            if inspect.isclass(operator_class_or_func):
                # 如果是类，实例化它
                operator_instance = operator_class_or_func(name, category)
            else:
                # 如果是函数，包装成 FunctionOperator
                operator_instance = FunctionOperator(
                    name, category, operator_class_or_func
                )

            self._operators[name] = operator_instance

            # 添加到分类
            if category not in self._categories:
                self._categories[category] = set()
            self._categories[category].add(name)

            # 处理别名
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = name

            return operator_instance

        return decorator

    def get(self, name: str) -> Optional["BaseOperator"]:
        """获取操作符"""
        # 检查别名
        actual_name = self._aliases.get(name, name)
        return self._operators.get(actual_name)

    def __getitem__(self, name: str) -> "BaseOperator":
        """支持 registry[name] 语法"""
        op = self.get(name)
        if op is None:
            raise KeyError(f"操作符 '{name}' 未注册")
        return op

    def __call__(self, name: str, *args, **kwargs):
        """支持直接调用"""
        return self[name](*args, **kwargs)

    def list_operators(self, category: str = None) -> List[str]:
        """列出操作符"""
        if category is None:
            return list(self._operators.keys())
        return list(self._categories.get(category, set()))

    def list_categories(self) -> List[str]:
        """列出所有分类"""
        return list(self._categories.keys())

    def batch_register_cast_ops(self, dtypes: List[str]):
        """批量注册类型转换操作符"""
        for dtype in dtypes:
            name = f"to_{dtype}"
            func = lambda x, dt=dtype: x.to(dtype=dt)
            self.register(name, "cast")(func)


# 全局注册器
operator_registry = OperatorRegistry()


class BaseOperator(ABC):
    """操作符基类"""

    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}', '{self.category}')"


class FunctionOperator(BaseOperator):
    """函数操作符包装器"""

    def __init__(self, name: str, category: str, func: Callable):
        super().__init__(name, category)
        self.func = func
        # 保留原函数的元信息
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class ArithmeticOperator(BaseOperator):
    """算术操作符"""

    def __init__(self, name: str, category: str, func: Callable = None):
        super().__init__(name, category)
        self.func = func

    def __call__(self, x, y):
        return self.func(x, y)


class UnaryOperator(BaseOperator):
    """一元操作符"""

    def __init__(self, name: str, category: str, func: Callable = None):
        super().__init__(name, category)
        self.func = func

    def __call__(self, x):
        return self.func(x)


# 使用装饰器注册操作符
@operator_registry.register("add", "arithmetic", aliases=["plus", "+"])
def add_op(x, y):
    """加法操作"""
    return operator.add(x, y)


@operator_registry.register("mul", "arithmetic", aliases=["multiply", "*"])
def mul_op(x, y):
    """乘法操作"""
    return operator.mul(x, y)


@operator_registry.register("sub", "arithmetic", aliases=["minus", "-"])
def sub_op(x, y):
    """减法操作"""
    return operator.sub(x, y)


@operator_registry.register("div", "arithmetic", aliases=["divide", "/"])
def div_op(x, y):
    """除法操作"""
    return operator.floordiv(x, y)


@operator_registry.register("min", "arithmetic")
def min_op(x, y):
    """最小值操作"""
    return x if x < y else y


@operator_registry.register("max", "arithmetic")
def max_op(x, y):
    """最大值操作"""
    return x if x >= y else y


@operator_registry.register("pos", "unary", aliases=["+"])
def pos_op(x):
    """正号操作"""
    return operator.pos(x)


@operator_registry.register("neg", "unary", aliases=["-"])
def neg_op(x):
    """负号操作"""
    return operator.neg(x)


@operator_registry.register("pass_through", "utility", aliases=["identity", "noop"])
def pass_through_op(x):
    """直通操作"""
    return x


# 批量注册类型转换操作符
dtypes = [
    "float32",
    "float16",
    "int32",
    "int64",
    "bool",
    "uint8",
    "int8",
    "int16",
    "uint16",
    "uint32",
    "uint64",
]
operator_registry.batch_register_cast_ops(dtypes)
