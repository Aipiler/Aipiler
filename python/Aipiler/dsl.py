from Aipiler.primitive import EinsumBuilder
from Aipiler.tensor import FakeTensor, FakeData, FakeScalar
from Aipiler.dim import Dim, dim
from Aipiler.basic_operator import operator_registry
from Aipiler.graph import EinsumGraph
from Aipiler.aot import export, CompiledModule
import inspect
import ast
import functools
from typing import Dict, List, Set, Tuple, Any, Optional, Callable, Union, Sequence
from types import FunctionType
import sys
import os
from copy import copy


def map(
    A: FakeData,
    B: FakeData,
    einsum_str: str,
    compute_op_str: str,
) -> FakeData:
    return EinsumBuilder.map(A, B, einsum_str, operator_registry.get(compute_op_str))


def reduce(
    A: FakeData,
    einsum_str: str,
    target_dim: Union[str, Sequence[str]],
    compute_op_str: str,
) -> FakeData:
    return EinsumBuilder.reduce(
        A, einsum_str, target_dim, operator_registry.get(compute_op_str)
    )


def unary(
    A: FakeData,
    unary_op_str: str,
) -> FakeData:
    assert isinstance(A, FakeTensor)
    dim_str = ""
    for i in range(len(A.symbolic_shapes)):
        dim_str += chr(ord("a") + i)
    return EinsumBuilder.unary(
        A, f"{dim_str} -> {dim_str}", operator_registry.get(unary_op_str)
    )


def cascade(func: FunctionType):
    R"""
    cascade is a black box / subgraph in einsum graph
    example (all map prims are element-wise add):
                (E)
                |                                  (E)
            |map|      # map3                     |
            /   \                             [cascade] ij,ij,ij,ij->ij
        |map|   |map|  # map1  map2  ==>     /   | |   \  
        /   \   /  \                        /   |   |   \ 
        (C)   |map|  (D) # map0             (C)  (A) (B)  (D)
                / \ 
            (A) (B)                             
    
    @einsum
    def test():
        ... # prepare A, B, C, D
        @cascade
        def four_add(A, B, C, D):
            t0 = map(A, B, "ij, ij -> ij", "+")
            t1 = map(C, t0, "ij, ij -> ij", "+")
            t2 = map(t0, D, "ij, ij -> ij", "+")
            t3 = map(t1, t2, "ij, ij -> ij", "+")
            return t3
        E = four_add(A, B, C, D)
        ...
    """
    debug = False
    if not einsum_env.initialized:
        einsum_env.initialize(debug)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # copy args
        cascade_inputs = args
        subgraph_inputs = []
        for cascade_input in cascade_inputs:
            if isinstance(cascade_input, FakeTensor):
                symbolic_shapes = []
                for d in cascade_input.symbolic_shapes:
                    symbolic_shapes.append(dim(d.size))
                subgraph_input = FakeTensor(
                    symbolic_shapes=symbolic_shapes,
                    dtype=cascade_input.dtype,
                    trace=None,
                )
            else:
                assert isinstance(cascade_input, FakeScalar)
                subgraph_input = copy(cascade_input)
            subgraph_inputs.append(subgraph_input)
        # run
        result = func(*subgraph_inputs, **kwargs)
        # trace graph
        subgraph = einsum_env.trace_from(result, inputs=subgraph_inputs)
        # create prim
        outputs = EinsumBuilder.cascade(
            *cascade_inputs, subgraph=subgraph, einsum_str=subgraph.summary_einsum_str()
        )
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

    return wrapper


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
        """
        TODO: trace einsumgraph with `inputs` and `outputs`. If inputs are None, trace until leaves of whole graph.
        for example:
        the whole graph is: t1=reduce(t0=map(A, B, ...), ...);
        if inputs = [t0], outputs = [t1], then the result of trace_from has nodes: [reduce]
        elif inputs= [A, B], outputs= [t1], then the result of trace_from has nodes: [reduce, map]
        """
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

        # print(f"开始执行函数 {func.__name__}，参数为 {args}, {kwargs}")
        result = func(*args, **kwargs)
        # print(f"结束执行函数 {func.__name__}")
        return result

    return wrapper


def compile_module(
    entry_point: functools,
    example_args: list[FakeTensor],
    *,
    dynamic: bool = False,
    save: bool = False,
    target_backend: str = "host",
) -> CompiledModule | None:
    """编译模块 - 统一的编译函数"""

    if dynamic:
        prefix = "dyn"
    else:
        prefix = "static"

    graph = einsum_env.compile(entry_point, example_args)
    exported = export(graph)

    if not save:
        return exported.compile(save_to=None, target_backend=target_backend)
    else:
        save_dir = f"einsum_{prefix}_vmfb"
        os.makedirs(save_dir, exist_ok=True)
        module_filepath = os.path.join(
            save_dir, f"{entry_point.__name__}_{prefix}_{target_backend}.vmfb"
        )
        exported.compile(save_to=module_filepath, target_backend=target_backend)
        return None
