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


from collections import OrderedDict
from Aipiler.tensor import Parameter
from typing import Iterable


class Module:
    """我们自己的Module基类，所有自定义模块都将继承它。"""

    def __init__(self):
        # 使用有序字典来保证模块和参数的顺序
        self._parameters = OrderedDict()
        self._children = OrderedDict()

    def register_parameter(self, name: str, param: Parameter):
        """显式注册一个参数。"""
        if not isinstance(param, Parameter):
            raise TypeError(
                f"Cannot assign torch.Tensor. Use MyParameter instead. (Got {type(param).__name__})"
            )
        self._parameters[name] = param

    def add_module(self, name: str, module):
        """显式注册一个子模块。"""
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{type(module).__name__} is not a MyBaseModule subclass")
        self._children[name] = module

    def __setattr__(self, name: str, value):
        """
        拦截所有属性分配，这是实现自动注册的核心！
        """
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.add_module(name, value)
        object.__setattr__(self, name, value)

    def named_parameters(self, prefix: str = ""):
        """
        递归地生成所有命名参数。这是一个生成器。
        """
        # 1. 先交出自己的直接参数
        for name, param in self._parameters.items():
            yield (prefix + "." + name if prefix else name, param)

        # 2. 递归地进入所有子模块
        for name, module in self._children.items():
            # 将子模块的名称加入前缀
            child_prefix = prefix + "." + name if prefix else name
            # 委派生成任务给子模块
            yield from module.named_parameters(prefix=child_prefix)

    def parameters(self):
        """一个方便的生成器，只返回参数张量本身。"""
        for _, param in self.named_parameters():
            yield param

    def __call__(self, *args, **kwargs):
        """允许我们像函数一样调用模块实例。"""
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """所有子类都应该重写这个方法。"""
        raise NotImplementedError

    def __repr__(self):
        # 创建一个漂亮的打印输出
        lines = [self.__class__.__name__ + "("]
        for name, module in self._children.items():
            lines.append(f"  ({name}): {repr(module)}")
        lines.append(")")
        return "\n".join(lines)


class ModuleList(Module):
    def __init__(self, modules: Iterable = None):
        super().__init__()
        if modules is not None:
            for module in modules:
                self.append(module)

    def append(self, module):
        # 使用从MyBaseModule继承来的add_module方法进行注册
        # 关键：使用数字字符串作为子模块的名称
        self.add_module(str(len(self._children)), module)
        return self

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self._children)
        return self._children[str(idx)]

    def __len__(self):
        return len(self._children)

    def __iter__(self):
        return iter(self._children.values())


def load_from_safetensor(model: Module, model_path: str):
    from safetensors.torch import load_file

    pytorch_state_dict = load_file(model_path)
    custom_named_params = dict(model.named_parameters())

    # 遍历从文件中加载的PyTorch权重
    for param_name, pytorch_tensor in pytorch_state_dict.items():
        print(f"正在处理参数: {param_name}...")

        # 检查这个参数是否存在于我们的自定义模型中
        if param_name in custom_named_params:
            # 获取自定义模型中对应的Parameter对象
            custom_param: Parameter = custom_named_params[param_name]

            # 检查形状是否匹配，这是一个非常重要的健全性检查
            if custom_param.numeric_shape == list(pytorch_tensor.shape):
                # 关键步骤：进行数据迁移
                custom_param._storage = pytorch_tensor  # 或者其他等效的赋值操作
                print(f"  ✅ 成功加载权重，形状: {pytorch_tensor.shape}")
            else:
                print(
                    f"  ❌ 形状不匹配！PyTorch: {pytorch_tensor.shape}, 自定义: {custom_param.dims()}"
                )
        else:
            print(f"  ⚠️ 警告: 在自定义模型中找不到参数 {param_name}")
    return model


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


def rearrange(
    A: Union[FakeTensor, List[FakeTensor]],
    pattern: str,
    **axes_length: Dict[str, int],
) -> FakeTensor:
    return EinsumBuilder.rearrange(A, pattern, **axes_length)


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
    print(graph)
    print("MLIR:")
    exported.print_readable()
    if not save:
        tmp = exported.compile(save_to=None, target_backend=target_backend)
        ctx = exported.mlir_context()
        del ctx
        return tmp
    else:
        save_dir = f"einsum_{prefix}_vmfb"
        os.makedirs(save_dir, exist_ok=True)
        module_filepath = os.path.join(
            save_dir, f"{entry_point.__name__}_{prefix}_{target_backend}.vmfb"
        )
        exported.compile(save_to=module_filepath, target_backend=target_backend)
        ctx = exported.mlir_context()
        del ctx
        return None
