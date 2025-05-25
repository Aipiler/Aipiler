from typing import Dict, List, Set, Tuple
from .einsumExpression import EinsumExpression
from .tensor import DataSpace, Rank, RankSet
from .rankVariable import RankVariable, RankVariableSet
from .rankExpression import RankMap, RankExpression, AffineRankExpression
from .einsumExpression import (
    MapEquation,
    ReduceEquation,
    PopulateEquation,
    UnaryEquation,
)
from .term import ConstTerm, VarTerm, AffineTerm
from .operators.compute import ComputeOperator
from .operators.coordinate import CoordinateOperator
from .operators.unary import UnaryOperator
from .manager import Context, Builder

import inspect
import ast
import functools
from typing import Dict, List, Set, Tuple, Any, Optional, Callable
import sys


# 全局环境配置
class EinsumEnvironment:
    """全局einsum操作环境配置"""

    def __init__(self):
        self.initialized = False
        self.debug_mode = False
        self.traced_functions = {}  # 存储已解析的函数
        self.context = None
        self.builder = None
        self.compiled_functions = set()  # 跟踪已编译的函数

    def get_builder(self) -> Builder:
        return self.builder

    def initialize(self, debug_mode=False):
        """初始化环境"""
        self.initialized = True
        self.context = Context()
        self.debug_mode = debug_mode
        if self.debug_mode:
            print("Einsum环境初始化完成，调试模式已启用")

    def add_traced_func(self, func: function, function_def: ast.FunctionDef):
        """添加要跟踪的函数"""
        if func.__name__ not in self.traced_functions:
            self.traced_functions[func.__name__] = (func, function_def)
            if self.debug_mode:
                print(f"函数 {func.__name__} 已添加到跟踪列表")

    def compile(self, entry_point=None):
        """
        编译einsum函数，但不执行结果

        Args:
            entry_point: 指定入口函数名，如果为None则编译所有函数
        """
        # 只编译einsum，不运行结果
        self.builder = Builder(self.context)
        # 根据output 从traced_functions中提取func，递归compile
        self.debug_mode = False

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


def parse_einsum_equation(
    equation: str,
) -> Tuple[List[List[str]], List[str], List[str]]:
    """
    解析einsum方程字符串，遵循严格的格式要求

    规则:
    1. 只支持一个或两个操作数（最多一个逗号）
    2. 必须有箭头'->'表示输出
    3. 如果是0维张量，必须用'_'表示，不允许空着不写
    4. 禁止省略号'...'
    5. 只接受小写字母和下划线作为下标

    Args:
        equation (str): 符合上述规则的einsum方程字符串

    Returns:
        Tuple[List[List[str]], List[str], List[str]]: 三元组，包含:
            - input_subscripts: 每个输入操作数的下标列表
            - output_subscripts: 输出的下标列表
            - sum_subscripts: 需要求和的下标列表
    """
    # 第一步：字符合法性检查
    # 创建包含所有允许字符的集合：小写字母a-z、下划线、逗号、箭头符号
    allowed_chars: Set[str] = set("abcdefghijklmnopqrstuvwxyz_,->")

    # 遍历方程中的每个字符，确保都在允许列表中
    for char in equation:
        if char not in allowed_chars:
            # 如果发现非法字符，抛出异常并指明是哪个字符
            raise ValueError(
                f"Einsum方程只能包含小写字母、下划线和方程结构符号，发现非法字符: '{char}'"
            )

    # 第二步：检查方程格式 - 必须有箭头
    # 箭头'->'是强制性的，用于分隔输入和输出部分
    if "->" not in equation:
        raise ValueError("Einsum方程必须包含箭头'->'来明确指定输出")

    # 第三步：拆分方程为输入和输出部分
    # 例如："ab,bc->ac" 分割为 "ab,bc" 和 "ac"
    input_part: str
    output_part: str
    input_part, output_part = equation.split("->")

    # 第四步：验证输出部分
    # 输出部分不能为空，0维标量必须用'_'表示而不是留空
    if not output_part:
        raise ValueError("输出部分不能为空，0维张量请使用'_'表示")

    # 第五步：验证操作数数量
    # 通过计算逗号数量来判断输入操作数数量，最多支持两个操作数（一个逗号）
    comma_count: int = input_part.count(",")
    if comma_count > 1:
        raise ValueError("只支持一个或两个操作数，发现多个逗号")

    # 第六步：解析输入操作数下标
    # 按逗号分割并转换为字符列表，例如："ab,bc" -> [['a','b'], ['b','c']]
    input_subscripts: List[List[str]] = [
        list(subscript) for subscript in input_part.split(",")
    ]

    # 第七步：验证每个操作数的下标
    # 确保每个操作数都有下标，0维张量必须用'_'而不是空字符串
    for i, subscripts in enumerate(input_subscripts):
        if not subscripts:
            raise ValueError(f"第{i+1}个操作数的下标不能为空，0维张量请使用'_'表示")

    # 第八步：统计每个下标在输入中的出现次数
    # 创建字典记录每个下标的出现频率，用于后续确定求和下标
    subscript_counts: Dict[str, int] = {}
    for subscript_list in input_subscripts:
        for subscript in subscript_list:
            # 如果下标已存在则加1，否则设为1
            subscript_counts[subscript] = subscript_counts.get(subscript, 0) + 1

    # 第九步：处理输出下标
    # 将输出部分转换为字符列表，例如："ac" -> ['a','c']
    output_subscripts: List[str] = list(output_part)

    # 第十步：验证输出下标合法性
    # 确保每个输出下标在输入中出现过（除了特殊的'_'标记）
    for subscript in output_subscripts:
        if subscript != "_" and subscript not in subscript_counts:
            # 如果输出中有未在输入中出现过的下标，报错
            raise ValueError(f"输出下标 '{subscript}' 未在任何输入操作数中出现")

    # 返回三个关键结果：
    # 1. 输入操作数的下标列表：例如 [['a','b'], ['b','c']]
    # 2. 输出的下标列表：例如 ['a','c']
    return input_subscripts, output_subscripts


def map(
    A: DataSpace,
    B: DataSpace,
    einsum_str: str,
    target_rank: list[str],
    computeOp: ComputeOperator,
) -> DataSpace:
    if einsum_env.debug_mode:
        print(f"执行map操作: {einsum_str}, 操作符: {computeOp}")
        # 在调试模式下执行实际操作
        # 这里需要您提供具体实现
        pass
    else:
        # 在编译模式下，将操作记录到Context中
        builder = einsum_env.get_builder()
        input_subscripts, output_subscripts = parse_einsum_equation(einsum_str)
        if len(input_subscripts) != 2:
            raise ValueError(
                f"len(input_subscripts)={len(input_subscripts)} isn't equal 2."
            )
        if len(output_subscripts) != 1:
            raise ValueError(
                f"len(output_subscripts)={len(output_subscripts)} isn't equal 1."
            )
        # TODO: 输入的A, B还是Tensor比较好，因为也需要维度确定。
        # TODO: 其实RankSet就是Tensor，因为确定了Rank的order和size。
        # TODO: Tensor可以归属于一个data space。
        # TODO: 因为存在reshape，transpose， slice这种操作，只改变layout，不改变数值。
        # TODO: 这种操作都可以用dataSpace与Tensor的抽象建模。
        builder.create_map_equation()


def reduce(
    A: DataSpace,
    einsum_str: str,
    target_rank: list[str],
    computeOp: ComputeOperator,
) -> DataSpace:
    pass


def populate(
    A: DataSpace,
    einsum_str: str,
    target_rank: list[str],
    computeOp: ComputeOperator,
    coordinateOp: CoordinateOperator,
) -> DataSpace:
    pass


def unary(
    A: DataSpace,
    einsum_str: str,
    unaryOp: UnaryOperator,
) -> DataSpace:
    pass


def extract_string_list(node):
    """从AST节点提取字符串列表"""
    result = []

    if isinstance(node, ast.List):
        for elt in node.elts:
            if hasattr(ast, "Constant") and isinstance(elt, ast.Constant):
                result.append(elt.value)
            elif isinstance(elt, ast.Str):  # Python 3.7及之前
                result.append(elt.s)

    return result


def handle_builtin_func(op_name: str, args, local_vars) -> DataSpace:
    result = None
    if op_name == "map":
        # 确保参数数量正确
        if len(args) < 5:
            raise ValueError(
                "map操作需要5个参数: A, B, einsum_str, target_ranks, computeOp"
            )

        # 提取参数
        a_name = args[0].id if isinstance(args[0], ast.Name) else None
        b_name = args[1].id if isinstance(args[1], ast.Name) else None

        # 处理不同AST版本
        if hasattr(ast, "Constant") and isinstance(args[2], ast.Constant):
            einsum_str = args[2].value
        elif isinstance(args[2], ast.Str):  # Python 3.7及之前
            einsum_str = args[2].s
        else:
            einsum_str = None

        # 提取目标秩列表
        target_ranks = extract_string_list(args[3])

        # 提取计算操作符
        if hasattr(ast, "Constant") and isinstance(args[4], ast.Constant):
            compute_op_str = args[4].value
        elif isinstance(args[4], ast.Str):  # Python 3.7及之前
            compute_op_str = args[4].s
        else:
            compute_op_str = None

        # 验证参数是否有效
        if None in (a_name, b_name, einsum_str, target_ranks, compute_op_str):
            raise ValueError("map操作的参数无效")

        # 执行map操作
        # TODO: computeOp的注册还要在context和builder中确定才行。
        compute_op = parse_compute_op(compute_op_str)
        result = map(
            local_vars[a_name], local_vars[b_name], einsum_str, target_ranks, compute_op
        )
    elif op_name == "reduce":
        # 确保参数数量正确
        if len(args) < 4:
            raise ValueError(
                "reduce操作需要4个参数: A, einsum_str, target_ranks, computeOp"
            )

        # 提取参数
        a_name = args[0].id if isinstance(args[0], ast.Name) else None

        # 提取einsum字符串
        if hasattr(ast, "Constant") and isinstance(args[1], ast.Constant):
            einsum_str = args[1].value
        elif isinstance(args[1], ast.Str):  # Python 3.7及之前
            einsum_str = args[1].s
        else:
            einsum_str = None

        # 提取目标秩列表
        target_ranks = extract_string_list(args[2])

        # 提取计算操作符
        if hasattr(ast, "Constant") and isinstance(args[3], ast.Constant):
            compute_op_str = args[3].value
        elif isinstance(args[3], ast.Str):  # Python 3.7及之前
            compute_op_str = args[3].s
        else:
            compute_op_str = None

        # 验证参数是否有效
        if None in (a_name, einsum_str, target_ranks, compute_op_str):
            raise ValueError("reduce操作的参数无效")

        # 执行reduce操作
        compute_op = parse_compute_op(compute_op_str)
        result = reduce(local_vars[a_name], einsum_str, target_ranks, compute_op)

    elif op_name == "populate":
        # 确保参数数量正确
        if len(args) < 5:
            raise ValueError(
                "populate操作需要5个参数: A, einsum_str, target_ranks, computeOp, coordinateOp"
            )

        # 提取参数
        a_name = args[0].id if isinstance(args[0], ast.Name) else None

        # 提取einsum字符串
        if hasattr(ast, "Constant") and isinstance(args[1], ast.Constant):
            einsum_str = args[1].value
        elif isinstance(args[1], ast.Str):  # Python 3.7及之前
            einsum_str = args[1].s
        else:
            einsum_str = None

        # 提取目标秩列表
        target_ranks = extract_string_list(args[2])

        # 提取操作符
        if hasattr(ast, "Constant") and isinstance(args[3], ast.Constant):
            compute_op_str = args[3].value
        elif isinstance(args[3], ast.Str):  # Python 3.7及之前
            compute_op_str = args[3].s
        else:
            compute_op_str = None

        if hasattr(ast, "Constant") and isinstance(args[4], ast.Constant):
            coordinate_op_str = args[4].value
        elif isinstance(args[4], ast.Str):  # Python 3.7及之前
            coordinate_op_str = args[4].s
        else:
            coordinate_op_str = None

        # 验证参数是否有效
        if None in (
            a_name,
            einsum_str,
            target_ranks,
            compute_op_str,
            coordinate_op_str,
        ):
            raise ValueError("populate操作的参数无效")

        # 执行populate操作
        compute_op = parse_compute_op(compute_op_str)
        coordinate_op = parse_coordinate_op(coordinate_op_str)
        rresult = populate(
            local_vars[a_name], einsum_str, target_ranks, compute_op, coordinate_op
        )

    elif op_name == "unary":
        # 确保参数数量正确
        if len(args) < 3:
            raise ValueError("unary操作需要3个参数: A, einsum_str, unaryOp")

        # 提取参数
        a_name = args[0].id if isinstance(args[0], ast.Name) else None

        # 提取einsum字符串
        if hasattr(ast, "Constant") and isinstance(args[1], ast.Constant):
            einsum_str = args[1].value
        elif isinstance(args[1], ast.Str):  # Python 3.7及之前
            einsum_str = args[1].s
        else:
            einsum_str = None

        # 提取操作符
        if hasattr(ast, "Constant") and isinstance(args[2], ast.Constant):
            unary_op_str = args[2].value
        elif isinstance(args[2], ast.Str):  # Python 3.7及之前
            unary_op_str = args[2].s
        else:
            unary_op_str = None

        # 验证参数是否有效
        if None in (a_name, einsum_str, unary_op_str):
            raise ValueError("unary操作的参数无效")

        # 执行unary操作
        unary_op = parse_unary_op(unary_op_str)
        result = unary(local_vars[a_name], einsum_str, unary_op)

    else:
        raise ValueError(f"不支持的操作类型: {op_name}")

    return result


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
        current_debug = einsum_env.debug_mode
        if debug:
            einsum_env.debug_mode = True

        if einsum_env.debug_mode:
            print(f"开始执行函数 {func.__name__}")

        # 绑定参数
        signature = inspect.signature(func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # 创建局部变量字典
        local_vars = dict(bound_args.arguments)

        try:

            # 标记结果变量
            result = None

            # 执行函数体中的每个语句
            for stmt in function_def.body:
                # 处理赋值语句
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                ):
                    target = stmt.targets[0].id

                    # 处理函数调用
                    if isinstance(stmt.value, ast.Call):
                        if isinstance(stmt.value.func, ast.Name):
                            func_name = stmt.value.func.id
                            func_args = stmt.value.args

                            # 处理基本DSL操作
                            if func_name in ("map", "reduce", "populate", "unary"):
                                local_vars[target] = handle_builtin_func(
                                    func_name, func_args, local_vars
                                )

                            # 处理被@einsum装饰的函数调用
                            elif func_name in einsum_env.traced_functions:
                                call_func, _ = einsum_env.traced_functions[func_name]
                                # 提取参数
                                call_args = []
                                for arg in stmt.value.args:
                                    if isinstance(arg, ast.Name):
                                        call_args.append(local_vars[arg.id])
                                    # 处理其他类型的参数...
                                local_vars[target] = call_func(*call_args)

                            # 其他普通函数调用
                            else:
                                pass

                # 处理返回语句
                elif isinstance(stmt, ast.Return):
                    if isinstance(stmt.value, ast.Name):
                        result = local_vars[stmt.value.id]
                        break
                    else:
                        raise ValueError("返回语句必须返回一个变量名")

            if einsum_env.debug_mode:
                print(
                    f"函数 {func.__name__} 执行完成，返回结果类型: {type(result).__name__}"
                )

            # 返回结果
            return result

        finally:
            # 恢复之前的调试设置
            einsum_env.debug_mode = current_debug

    return wrapper


# 使用示例
"""
@einsum
def foo(A: Tensor, B: Tensor) -> Tensor:
    C = map(A, B, "k, k -> k", ["k"], "*")
    X = reduce(C, "k -> _", ["k"], "+")
    Y = reduce(A, "k -> _", ["k"], "+")
    Z = map(X, Y, "_, _ -> _", ["_"], "*")
    return Z
"""

"""
1. expression


def foo() {
    
}


def matmul(A: Tensor, B: Tensor) -> Tensor:
    Z = foo(A, B)
    X = einsum.map(A, B, "ik, kj -> ikj", ["k"], "*")
    C = eisnum.reduce(X, "ikj -> ij", ["k"], "+")
    return C
    
matmul()
2. analyze & fusion: some passes

@einsum
def matmul(A: Tensor, B: Tensor) -> Tensor:
    C = einsum.cascade(A, B, ["i", "j", "k"]){
        X = map(A, B, "ik, kj -> ikj", ["k"], "*")
        C = reduce(X, "ikj -> ij", ["k"], "+")
    }
    return C  
     
3. rewrite (some schedules? or some passes?)

@einsum
def matmul(A: Tensor, B: Tensor) -> Tensor:
    C = einsum.fusion(A, B, ["i", "j", "k"]){
        X = map(A, B, "ik, kj -> ikj", ["k"], "*")
        C = reduce(X, "ikj -> ij", ["k"], "+")
    }
    return C  
"""


def test_dsl():

    @einsum
    def foo():
        pass

    einsum_env.compile(entry_point=foo)
    einsum_env.debug_exec()


def test_parse_einsum_equation():

    def explain_einsum_operation(equation):
        """
        解释einsum方程表示的操作

        Args:
            equation (str): einsum方程字符串

        Returns:
            str: 操作的文字解释
        """
        try:
            input_subscripts, output_subscripts, sum_subscripts = parse_einsum_equation(
                equation
            )
        except ValueError as e:
            return f"方程格式错误: {str(e)}"

        explanation = []
        explanation.append(f"Einsum方程 '{equation}' 解析结果：")
        explanation.append(f"- 输入操作数下标: {input_subscripts}")
        explanation.append(f"- 输出下标: {output_subscripts}")
        explanation.append(f"- 求和下标: {sum_subscripts}")

        # 添加一些常见模式的识别
        if len(input_subscripts) == 2:
            if equation == "ij,jk->ik":
                explanation.append("- 此操作是标准矩阵乘法：A @ B")
            elif equation == "ij,ij->ij":
                explanation.append("- 此操作是逐元素乘法：A * B")
            elif equation == "ij,ij->i":
                explanation.append(
                    "- 此操作是沿着第二个维度求和的点积：torch.sum(A * B, dim=1)"
                )
            elif equation == "ij,ji->_":
                explanation.append(
                    "- 此操作是元素乘积的总和（标量积）：torch.sum(A * B.T)"
                )

        if len(input_subscripts) == 1:
            if equation == "ii->_":
                explanation.append("- 此操作是计算矩阵的迹：torch.trace(A)")
            elif equation == "ii->i":
                explanation.append("- 此操作是提取矩阵的对角线元素：torch.diagonal(A)")
            elif equation == "ij->ji":
                explanation.append("- 此操作是矩阵转置：A.T")

        return "\n".join(explanation)

    # 示例：测试解析器
    examples = [
        "ii->_",  # 矩阵的迹
        "ii->i",  # 矩阵对角线元素
        "ij,jk->ik",  # 矩阵乘法
        "ij,ik->jk",  # 一种转置缩并操作
        "i,j->ij",  # 外积
        "ij,ji->_",  # 标量积
        "ij,ij->ij",  # 哈达玛积（逐元素乘法）
        "_,i->i",  # 标量乘以向量
        "i->_",  # 向量的总和
    ]

    # 演示解析过程
    for eq in examples:
        print("=" * 50)
        print(explain_einsum_operation(eq))
        print()

    # 验证规则检查功能
    invalid_examples = [
        "IJ,JK->IK",  # 使用大写字母
        "i*j->ij",  # 使用非法字符
        "ij,jk,kl->il",  # 超过两个操作数
        "ij,jk",  # 没有箭头
        "ij,jk->",  # 输出为空
        "ij->ik",  # 输出中的下标在输入中不存在
        "ij,->ij",  # 空操作数
        "...ij->ij",  # 使用省略号
    ]

    print("=" * 50)
    print("测试非法输入:")
    for invalid_eq in invalid_examples:
        explain_result = explain_einsum_operation(invalid_eq)
        if explain_result.startswith("方程格式错误"):
            print(f"验证成功，{invalid_eq}: {explain_result}")
        else:
            print(f"未能捕获非法输入: {invalid_eq}")

    # 详细分析特定示例
    detailed_example = "ab,bc->ac"  # 标准矩阵乘法
    try:
        input_subscripts, output_subscripts, sum_subscripts = parse_einsum_equation(
            detailed_example
        )

        print("=" * 50)
        print(f"详细分析 '{detailed_example}':")
        print(f"1. 分离操作数：")
        print(f"   - 第一个操作数的下标：{input_subscripts[0]} (形状如 [d_a, d_b])")
        print(f"   - 第二个操作数的下标：{input_subscripts[1]} (形状如 [d_b, d_c])")
        print(f"2. 确定输出下标：{output_subscripts} (形状将是 [d_a, d_c])")
        print(f"3. 确定求和下标：{sum_subscripts} (维度 b 将被求和)")
        print(
            f"4. 操作描述：这是标准矩阵乘法 A @ B，我们沿着A的列和B的行（共享维度b）求和"
        )
    except ValueError as e:
        print(f"解析 '{detailed_example}' 失败: {e}")


if __name__ == "__main__":
    test_dsl()
