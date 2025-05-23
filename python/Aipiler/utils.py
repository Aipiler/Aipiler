import re
from typing import Tuple, List, Set, Dict
# from Aipiler.rank import *

def parse_einsum_str(
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
    equation = equation.replace(" ", "")
    allowed_chars: Set[str] = set("abcdefghijklmnopqrstuvwxyz_,->+*")
    
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
    # input_subscripts: List[List[str]] = [
    #     list(subscript) for subscript in input_part.split(",")
    # ]
    input_subscripts = input_part.split(",")
    
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

# print(parse_einsum_str("ik,kj->ij"))
print(parse_einsum_str("q+s, s -> qs"))

