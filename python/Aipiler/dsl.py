from Aipiler.primitive import EinsumBuilder
from Aipiler.tensor import FakeTensor
from Aipiler.dim import Dim, create_dim
from Aipiler.basic_operator import operator_registry
from Aipiler.graph import EinsumGraph
import inspect
import ast
import functools
from typing import Dict, List, Set, Tuple, Any, Optional, Callable, Union, Sequence
import sys


def map(
    A: FakeTensor,
    B: FakeTensor,
    einsum_str: str,
    target_dim: list[str],
    compute_op_str: str,
) -> FakeTensor:
    return EinsumBuilder.map(
        A, B, einsum_str, target_dim, operator_registry.get(compute_op_str)
    )


def reduce(
    A: FakeTensor,
    einsum_str: str,
    target_dim: list[str],
    compute_op_str: str,
) -> FakeTensor:
    return EinsumBuilder.reduce(
        A, einsum_str, target_dim, operator_registry.get(compute_op_str)
    )


def unary(
    A: FakeTensor,
    unary_op_str: str,
) -> FakeTensor:
    return EinsumBuilder.unary(A, operator_registry.get(unary_op_str))


# 全局环境配置
class EinsumEnvironment:
    """全局einsum操作环境配置"""

    def __init__(self):
        self.initialized = False
        self.debug_mode = False
        self.traced_functions = {}  # 存储已解析的函数
        self.compiled_functions = set()  # 跟踪已编译的函数

    def initialize(self, debug_mode=False):
        """初始化环境"""
        self.initialized = True
        self.debug_mode = debug_mode
        if self.debug_mode:
            print("Einsum环境初始化完成，调试模式已启用")

    def add_traced_func(self, func: functools, function_def: ast.FunctionDef):
        """添加要跟踪的函数"""
        if func.__name__ not in self.traced_functions:
            self.traced_functions[func.__name__] = (func, function_def)
            if self.debug_mode:
                print(f"函数 {func.__name__} 已添加到跟踪列表")

    def compile(
        self, entry_point: functools, example_args: Sequence[FakeTensor]
    ) -> EinsumGraph:
        """
        编译einsum函数，但不执行结果

        Args:
            entry_point: 指定入口函数名，如果为None则编译所有函数
        """
        outputs = entry_point(*example_args)
        return self.trace_from(outputs, inputs=example_args)

    def trace_from(
        self,
        outputs: Optional[Union[FakeTensor, Sequence[FakeTensor]]],
        inputs: Optional[Union[FakeTensor, Sequence[FakeTensor]]] = None,
    ) -> EinsumGraph:

        if isinstance(outputs, FakeTensor):
            if outputs._trace is None:
                raise ValueError("trace_from expects symbol tensor(s)")
            outputs = [outputs]
        else:
            outputs = list(outputs)
            assert all(isinstance(v, FakeTensor) for v in outputs)

        if inputs is not None:
            if isinstance(inputs, FakeTensor):
                inputs = [inputs]
            else:
                inputs = list(inputs)
        return EinsumGraph(outputs, inputs).update_nodes()

    def debug_exec(self):
        # 不进行代码生成，直接在python环境运行结果
        self.debug_mode = True
        # 根据input开始，顺序调用traced_functions中的func，执行结果

    def reset(self):
        """重置环境"""
        self.initialized = False
        self.debug_mode = False
        self.traced_functions.clear()
        self.context = None
        self.builder = None


# 创建一个全局环境实例
einsum_env = EinsumEnvironment()


def initialize_einsum_environment(debug_mode=False):
    """显式初始化einsum环境的辅助函数（可选使用）"""
    if einsum_env.initialized:
        einsum_env.reset()  # 重置现有环境
    einsum_env.initialize(debug_mode)


def einsum(func, *, debug=False):
    """
    用于解析和执行einsum DSL代码的装饰器

    支持以下操作:
    - map(A, B, "einsum_str", ["target_ranks"], "op")
    - reduce(A, "einsum_str", ["target_ranks"], "op")
    - populate(A, "einsum_str", ["target_ranks"], "computeOp", "coordinateOp")
    - unary(A, "einsum_str", "unaryOp")

    示例:
    @einsum
    def foo(A: DataSpace, B: DataSpace) -> DataSpace:
        C = map(A, B, "k, k -> k", ["k"], "*")
        X = reduce(C, "k -> _", ["k"], "+")
        Y = reduce(A, "k -> _", ["k"], "+")
        Z = map(X, Y, "_, _ -> _", ["_"], "*")
        return Z
    """

    # 支持两种使用方式: @einsum 和 @einsum(debug=True)
    if func is None:
        return lambda f: einsum(f, debug=debug)

    # 自动初始化环境（如果尚未初始化）
    if not einsum_env.initialized:
        einsum_env.initialize(debug)
        if debug:
            print("Einsum环境已自动初始化")

    # 获取函数签名和源代码
    source = inspect.getsource(func)
    tree = ast.parse(source)
    function_def = tree.body[0]

    # 存储已解析的函数AST，用于支持嵌套函数调用
    einsum_env.add_traced_func(func=func, function_def=function_def)

    # 创建转换后的函数
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 记录当前运行环境，用于支持嵌套调用

        print(f"开始执行函数 {func.__name__}，参数为 {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"结束执行函数 {func.__name__}")
        return result

    return wrapper
