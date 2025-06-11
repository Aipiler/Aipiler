from functools import wraps
from Aipiler.lang import *
import inspect
from typing import Callable, Dict, List, Optional, Set


class ComputeOperator:
    """
    包装一个已注册的操作符。
    这个类统一了所有操作符的表示，其核心是持有一个可直接调用的MLIR操作符对象。
    """

    def __init__(
        self, name: str, category: str, op_callable: Callable, doc: Optional[str] = None
    ):
        self.name = name
        self.category = category
        self.op_callable = op_callable  # 直接持有 MLIR 的可调用对象, e.g., BinaryFn.add
        self.__doc__ = doc if doc else getattr(op_callable, "__doc__", "")
        self.__name__ = name

    def __call__(self, *args, **kwargs):
        """
        当操作符被调用时，直接执行底层的 MLIR 可调用对象。
        例如: registry('add', tensor1, tensor2) -> self.op_callable(tensor1, tensor2)
        """
        return self.op_callable(*args, **kwargs)

    def get_op_callable(self) -> Callable:
        return self.op_callable

    def __repr__(self):
        return f"ComputeOperator(name='{self.name}', category='{self.category}')"


class OperatorRegistry:
    """
    操作符注册中心。
    用于存储、分类和调用映射到 MLIR DSL 的各种算子。
    """

    def __init__(self):
        self._operators: Dict[str, ComputeOperator] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, name: str, category: str, aliases: Optional[List[str]] = None):
        """
        注册一个 MLIR 操作符的装饰器。

        用法:
            被装饰的函数应该不带任何参数，并返回一个可调用的 MLIR 操作符对象。
            这个函数本身作为一个“工厂”，主要用于提供注册元数据（如文档字符串）。

            @operator_registry.register("add", "binary")
            def add():
                \"\"\"这是一个加法操作\"\"\"
                return BinaryFn.add
        """
        aliases = aliases or []

        def decorator(factory_func: Callable[[], Callable]):
            # 1. 调用工厂函数，获取真正的 MLIR 操作符对象 (e.g., BinaryFn.add)
            op_callable = factory_func()

            # 2. 创建并存储我们统一的 ComputeOperator 实例
            operator_instance = ComputeOperator(
                name, category, op_callable, doc=factory_func.__doc__
            )

            if name in self._operators:
                raise ValueError(f"操作符 '{name}' 已被注册。")
            self._operators[name] = operator_instance

            # 3. 添加到分类
            self._categories.setdefault(category, set()).add(name)

            # 4. 处理别名
            for alias in aliases:
                if alias in self._aliases:
                    raise ValueError(f"别名 '{alias}' 已被注册。")
                self._aliases[alias] = name

            return operator_instance

        return decorator

    def get(self, name: str) -> Optional[ComputeOperator]:
        """通过名称或别名获取已注册的操作符。"""
        actual_name = self._aliases.get(name, name)
        return self._operators.get(actual_name)

    def __getitem__(self, name: str) -> ComputeOperator:
        """支持 registry[name] 语法，找不到则抛出 KeyError。"""
        op = self.get(name)
        if op is None:
            raise KeyError(f"操作符 '{name}' 或其别名未注册。")
        return op

    def __call__(self, name: str, *args, **kwargs):
        """支持 registry(name, *args, **kwargs) 的直接调用语法。"""
        return self[name](*args, **kwargs)

    def list_operators(self, category: Optional[str] = None) -> List[str]:
        """列出已注册的操作符名称。"""
        if category:
            return sorted(list(self._categories.get(category, set())))
        return sorted(list(self._operators.keys()))

    def list_categories(self) -> List[str]:
        """列出所有分类。"""
        return sorted(list(self._categories.keys()))


# --- 全局注册器实例 ---
operator_registry = OperatorRegistry()

# ===================================================================
#             使用新的注册方式注册所有 MLIR 操作符
# ===================================================================


#
# 注册二元操作符 (Binary Operators)
#
@operator_registry.register("add", category="binary", aliases=["+", "plus"])
def add():
    """二元加法，映射到 linalg/arith dialect 的 `add`。"""
    return BinaryFn.add


@operator_registry.register("sub", category="binary", aliases=["-", "minus"])
def sub():
    """二元减法，映射到 linalg/arith dialect 的 `sub`。"""
    return BinaryFn.sub


@operator_registry.register("mul", category="binary", aliases=["*", "multiply"])
def mul():
    """二元乘法，映射到 linalg/arith dialect 的 `mul`。"""
    return BinaryFn.mul


@operator_registry.register("div", category="binary", aliases=["/"])
def div():
    """二元有符号整数或浮点除法，映射到 `arith.divsi` 或 `arith.divf`。"""
    return BinaryFn.div


@operator_registry.register("div_unsigned", category="binary")
def div_unsigned():
    """二元无符号整数除法，映射到 `arith.divui`。"""
    return BinaryFn.div_unsigned


@operator_registry.register("max", category="binary", aliases=["max_signed"])
def max_signed():
    """二元有符号最大值，映射到 `arith.maxsi` 或 `arith.maximumf`。"""
    return BinaryFn.max_signed


@operator_registry.register("min", category="binary", aliases=["min_signed"])
def min_signed():
    """二元有符号最小值，映射到 `arith.minsi` 或 `arith.minimumf`。"""
    return BinaryFn.min_signed


@operator_registry.register("max_unsigned", category="binary")
def max_unsigned():
    """二元无符号整数最大值，映射到 `arith.maxui`。"""
    return BinaryFn.max_unsigned


@operator_registry.register("min_unsigned", category="binary")
def min_unsigned():
    """二元无符号整数最小值，映射到 `arith.minui`。"""
    return BinaryFn.min_unsigned


@operator_registry.register("powf", category="binary", aliases=["^"])
def powf():
    """二元浮点数幂运算，映射到 `math.powf`。"""
    return BinaryFn.powf


#
# 注册一元操作符 (Unary Operators)
#
@operator_registry.register("exp", category="unary")
def exp():
    """一元指数运算 (e^x)，映射到 `math.exp`。"""
    return UnaryFn.exp


@operator_registry.register("log", category="unary")
def log():
    """一元自然对数，映射到 `math.log`。"""
    return UnaryFn.log


@operator_registry.register("abs", category="unary")
def abs():
    """一元浮点数绝对值，映射到 `math.absf`。"""
    return UnaryFn.abs


@operator_registry.register("ceil", category="unary")
def ceil():
    """一元向上取整，映射到 `math.ceil`。"""
    return UnaryFn.ceil


@operator_registry.register("floor", category="unary")
def floor():
    """一元向下取整，映射到 `math.floor`。"""
    return UnaryFn.floor


@operator_registry.register("neg", category="unary", aliases=["negf"])
def neg():
    """一元取反 (浮点数)，映射到 `arith.negf`。"""
    return UnaryFn.negf


@operator_registry.register("reciprocal", category="unary")
def reciprocal():
    """一元求倒数 (1/x)。"""
    return UnaryFn.reciprocal


@operator_registry.register("round", category="unary")
def round():
    """一元四舍五入，映射到 `math.round`。"""
    return UnaryFn.round


@operator_registry.register("sqrt", category="unary")
def sqrt():
    """一元平方根，映射到 `math.sqrt`。"""
    return UnaryFn.sqrt


@operator_registry.register("rsqrt", category="unary")
def rsqrt():
    """一元平方根倒数 (1/√x)，映射到 `math.rsqrt`。"""
    return UnaryFn.rsqrt


@operator_registry.register("square", category="unary")
def square():
    """一元平方 (x*x)。"""
    return UnaryFn.square


@operator_registry.register("tanh", category="unary")
def tanh():
    """一元双曲正切，映射到 `math.tanh`。"""
    return UnaryFn.tanh


@operator_registry.register("erf", category="unary")
def erf():
    """一元高斯误差函数，映射到 `math.erf`。"""
    return UnaryFn.erf


#
# 注册三元操作符 (Ternary Operators)
#
@operator_registry.register("select", category="ternary")
def select():
    """三元选择操作，映射到 `arith.select`。"""
    return TernaryFn.select


#
# 注册类型转换操作符 (Type Casting Operators)
#
@operator_registry.register("cast_signed", category="cast")
def cast_signed():
    """有符号类型转换，例如 `arith.extsi`。"""
    return TypeFn.cast_signed


@operator_registry.register("cast_unsigned", category="cast")
def cast_unsigned():
    """无符号类型转换，例如 `arith.extui`。"""
    return TypeFn.cast_unsigned


# # ===================================================================
# #             注册所有归约操作符 (Reduction Operators)
# # ===================================================================


# @operator_registry.register("reduce_add", category="reduction")
# def reduce_add():
#     """
#     加法归约。
#     用法: operator_registry('reduce_add')[dim0, dim1](tensor)
#     """
#     return ReduceFn.add


# @operator_registry.register("reduce_mul", category="reduction")
# def reduce_mul():
#     """
#     乘法归约。
#     用法: operator_registry('reduce_mul')[dim0, dim1](tensor)
#     """
#     return ReduceFn.mul


# @operator_registry.register(
#     "reduce_max", category="reduction", aliases=["reduce_max_signed"]
# )
# def reduce_max_signed():
#     """
#     有符号最大值归约。
#     用法: operator_registry('reduce_max')[dim0, dim1](tensor)
#     """
#     return ReduceFn.max_signed


# @operator_registry.register(
#     "reduce_min", category="reduction", aliases=["reduce_min_signed"]
# )
# def reduce_min_signed():
#     """
#     有符号最小值归约。
#     用法: operator_registry('reduce_min')[dim0, dim1](tensor)
#     """
#     return ReduceFn.min_signed


# @operator_registry.register("reduce_max_unsigned", category="reduction")
# def reduce_max_unsigned():
#     """
#     无符号最大值归约。
#     用法: operator_registry('reduce_max_unsigned')[dim0, dim1](tensor)
#     """
#     return ReduceFn.max_unsigned


# @operator_registry.register("reduce_min_unsigned", category="reduction")
# def reduce_min_unsigned():
#     """
#     无符号最小值归约。
#     用法: operator_registry('reduce_min_unsigned')[dim0, dim1](tensor)
#     """
#     return ReduceFn.min_unsigned
