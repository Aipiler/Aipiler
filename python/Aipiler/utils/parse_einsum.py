from typing import Tuple, List, Set, Dict
import functools


def parse_subscripts_part(part: str):
    # "ab,bc" -> ['ab', 'bc']
    part_subscripts = part.split(",")
    # 确保每个操作数都有下标，0维张量必须用'_'而不是空字符串
    for i, subscripts in enumerate(part_subscripts):
        if not subscripts:
            raise ValueError(f"第{i+1}个操作数的下标不能为空，0维张量请使用'_'表示")

    # ['ab', 'bc'] -> [['a', 'b'], ['b', 'c']]
    rets = []
    for part_subscript in part_subscripts:
        part_subscript_list = list(part_subscript)
        for idx in part_subscript_list:
            if idx == "_" and len(part_subscript_list) != 1:
                raise ValueError(f"错误的输入{part_subscript}")
        rets.append(part_subscript_list)
    return rets


def parse_einsum_str(
    equation: str,
) -> Tuple[List[List[str]], List[List[str]]]:
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
        Tuple[List[List[str]], List[List[str]]]: 二元组，包含:
            - input_subscripts:  每个输入操作数的下标列表
            - output_subscripts: 每个输出的下标列表
    """
    # 第一步：字符合法性检查
    # 创建包含所有允许字符的集合：小写字母a-z、下划线、逗号、箭头符号
    # TODO: support affine expression
    equation = equation.replace(" ", "")
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
    inputs_part: str
    outputs_part: str
    inputs_part, outputs_part = equation.split("->")
    inputs = parse_subscripts_part(inputs_part)
    outputs = parse_subscripts_part(outputs_part)

    # 第十步：验证输出下标合法性
    # 确保每个输出下标在输入中出现过（除了特殊的'_'标记）
    subscript_counts = []
    for input_subscript in inputs:
        subscript_counts += input_subscript

    for subscript_list in outputs:
        for sp in subscript_list:
            if sp != "_" and sp not in subscript_counts:
                # 如果输出中有未在输入中出现过的下标，报错
                raise ValueError(f"输出下标 '{sp}' 未在任何输入操作数中出现")
    return inputs, outputs
