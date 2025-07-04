from Aipiler.tensor import FakeTensor, FakeData, FakeScalar
from Aipiler.basic_operator import ComputeOperator
from Aipiler.dim import Dim, dims
from typing import (
    List,
    Union,
    Sequence,
    Dict,
    Any,
    overload,
    Callable,
    Tuple,
    Set,
    Optional,
)
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from copy import copy
from bidict import bidict
from Aipiler.axis import Axis
from enum import Enum
import re
import __future__


class DimAxisRelation:
    def __init__(self):
        self._dim_to_axis: Dict[Dim, Axis] = dict()
        self._axis_to_dim: Dict[Axis, Dim] = dict()

    @property
    def dims(self):
        return list(self._dim_to_axis.keys())

    @property
    def axes(self):
        return list(self._axis_to_dim.keys())

    def insert(self, dim: Dim, axis: Axis):
        if dim in self._dim_to_axis or axis in self._axis_to_dim:
            raise ValueError(f"Pair {dim} - {axis} already existed.")
        self._dim_to_axis[dim] = axis
        self._axis_to_dim[axis] = dim

    def __getitem__(self, key) -> Union[Dim, Axis]:
        if isinstance(key, Dim) and key in self._dim_to_axis:
            return self._dim_to_axis[key]
        elif isinstance(key, Axis) and key in self._axis_to_dim:
            return self._axis_to_dim[key]
        else:
            raise KeyError(f"KeyError: {key}")


class AxisAxisRelation:
    class RelationShip(Enum):
        EQ = "Equality between inputs or outputs"
        DEPEND_EQ = "Dependency and equality between input and output"
        DEPEND_SPLIT = "Split one input axis into multi output axes"
        DEPEND_MERGE = "Merge multi input axes into one output axes"

    def __init__(self, inputs_axes: List[List[Axis]], outputs_axes: List[List[Axis]]):
        self._inputs_axes = inputs_axes
        self._outputs_axes = outputs_axes
        self.all_input_axes = [_ for __ in inputs_axes for _ in __]
        self.all_output_axes = [_ for __ in outputs_axes for _ in __]

        # EQ: equal axes, including axes in DEPEND_EQ
        # key: any axis; value: equal axes
        self.equal_dict: Dict[Axis, Set[Axis]] = dict()

        # DEPEND_EQ
        # key: input axis, value: equal output axes
        self.depend_eq_dict_key_input: Dict[Axis, Set[Axis]] = dict()
        # key: output axis, value: equal input axes
        self.depend_eq_dict_key_output: Dict[Axis, Set[Axis]] = dict()

        # DEPEND_SPLIT
        # key: input axis, value: output axes
        self.depend_split_dict_key_input: Dict[Axis, Set[Axis]] = dict()
        # key: output axis, value: splited input axis
        self.depend_split_dict_key_output: Dict[Axis, Axis] = dict()

        # DEPEND_MERGE
        # key: input axis, value: merged output axis
        self.depend_merge_dict_key_input: Dict[Axis, Axis] = dict()
        # key: output axis, value: input axes
        self.depend_merge_dict_key_output: Dict[Axis, Set[Axis]] = dict()

    @property
    def inputs_axes(self):
        return self._inputs_axes

    @property
    def outputs_axes(self):
        return self._outputs_axes

    @overload
    def insert(self, i: Axis, o: Axis, relationship: "AxisAxisRelation.RelationShip"):
        """Insert axes with relationship.

        Args:
            i (Axis): input axis
            o (Axis): output axis
            relationship (RelationShip): relationship between i and o
        """
        ...

    @overload
    def insert(self, *axes, relationship: RelationShip = RelationShip.EQ):
        """Insert equal axes.

        Args:
            relationship (RelationShip, optional): _description_. Defaults to RelationShip.EQ.
        """
        ...

    def insert(self, *args, **kwargs):
        if "relationship" in kwargs:
            axes = args
            relation = kwargs["relationship"]
        elif isinstance(args[-1], AxisAxisRelation.RelationShip):
            axes = args[:-1]
            relation = args[-1]
        else:
            axes = args
            relation = AxisAxisRelation.RelationShip.EQ
        if not isinstance(relation, AxisAxisRelation.RelationShip):
            raise ValueError("Expected relationship in function insert")

        if relation == AxisAxisRelation.RelationShip.EQ:
            old_set: Set[Axis] = None
            for axis in axes:
                if axis in self.equal_dict:
                    old_set = self.equal_dict[axis]
                    break
            if old_set is None:
                old_set = set(axes)
            else:
                old_set.update(axes)
            for axis in old_set:
                self.equal_dict[axis] = old_set
        elif relation == AxisAxisRelation.RelationShip.DEPEND_EQ:
            assert len(axes) == 2
            i, o = axes
            # update depend_eq_dict_key_input
            if i in self.depend_eq_dict_key_input:
                self.depend_eq_dict_key_input[i].add(o)
            else:
                self.depend_eq_dict_key_input[i] = {o}
            # update depend_eq_dict_key_output
            if o in self.depend_eq_dict_key_output:
                self.depend_eq_dict_key_output[o].add(i)
            else:
                self.depend_eq_dict_key_output[o] = {i}

        elif relation == AxisAxisRelation.RelationShip.DEPEND_SPLIT:
            assert len(axes) == 2
            i, o = axes
            # update depend_split_dict_key_input
            if i in self.depend_split_dict_key_input:
                self.depend_split_dict_key_input[i].add(o)
            else:
                self.depend_split_dict_key_input[i] = {o}
            # update depend_split_dict_key_output
            if (
                o in self.depend_split_dict_key_output
                and self.depend_split_dict_key_output[o] is not i
            ):
                raise ValueError(f"Wrong Split relation between {i} and {o}")
            else:
                self.depend_split_dict_key_output[o] = i

        elif relation == AxisAxisRelation.RelationShip.DEPEND_MERGE:
            assert len(axes) == 2
            i, o = axes
            # update depend_merge_dict_key_input
            if (
                i in self.depend_merge_dict_key_input
                and self.depend_merge_dict_key_input[i] is not o
            ):
                raise ValueError(f"Wrong merge relation between {i} and {o}")
            else:
                self.depend_merge_dict_key_input[i] = o
            # update depend_merge_dict_key_output
            if o in self.depend_merge_dict_key_output:
                self.depend_merge_dict_key_output[o].add(i)
            else:
                self.depend_merge_dict_key_output[o] = {i}
        else:
            assert False

    def __repr__(self):
        return "equal = \t\t{},\ndependent&equal = \t{},\ndependent&spliting = \t{},\ndependent&merging = \t{}".format(
            self.equal_dict,
            self.depend_eq_dict_key_input,
            self.depend_split_dict_key_input,
            self.depend_merge_dict_key_input,
        )


def parse_einsum_str(equation: str) -> AxisAxisRelation:
    def no_nest_parenthesis(part: str) -> bool:
        # check no nested parenthesis
        need_another = False
        for l in part:
            if l == "(":
                if need_another:
                    return False
                else:
                    need_another = True

            elif l == ")":
                if need_another:
                    need_another = False
                else:
                    raise ValueError("Unbalanced parentheses.")
        return True

    def parse_single(
        part: str, is_input: bool, idx: Optional[int] = None
    ) -> List[Axis]:
        # parse str like "ab", "a(bc)d"
        if not no_nest_parenthesis(part):
            raise ValueError(f"Nested parentheses in {part}")

        # \((.*?)\) 是第一个捕获组，(.) 是第二个捕获组 (这里为了简单直接用 . 匹配所有单个字符)
        pattern = r"\((.*?)\)|(.)"

        # re.findall会返回一个元组列表，因为有两个捕获组
        # 例如: [('', 'a'), ('', 'b'), ('abc', ''), ...]
        matches = re.findall(pattern, part)

        # 对于每个元组 (group1, group2)，取非空的那一个
        names = [group1 or group2 for group1, group2 in matches]
        result = [
            Axis(name, is_input, idx_inside, idx)
            for idx_inside, name in enumerate(names)
        ]
        return result

    def parse_subscripts_part(part: str, is_input: bool):
        # "ab,bc" -> ['ab', 'bc']
        part_subscripts = part.split(",")
        # 确保每个操作数都有下标，0维张量必须用'_'而不是空字符串
        for i, subscripts in enumerate(part_subscripts):
            if not subscripts:
                raise ValueError(f"第{i+1}个操作数的下标不能为空，0维张量请使用'_'表示")
        axes: List[List[Axis]] = []
        for idx, part in enumerate(part_subscripts):
            io_axes = parse_single(part, is_input, idx)
            axes.append(io_axes)

        return axes

    equation = equation.replace(" ", "")
    allowed_chars: Set[str] = set("abcdefghijklmnopqrstuvwxyz_(),->")

    for char in equation:
        if char not in allowed_chars:
            raise ValueError(
                f"Einsum方程只能包含小写字母、下划线和方程结构符号，发现非法字符: '{char}'"
            )

    if "->" not in equation:
        raise ValueError("Einsum方程必须包含箭头'->'来明确指定输出")

    # 拆分方程为输入和输出部分
    inputs_part: str
    outputs_part: str
    inputs_part, outputs_part = equation.split("->")

    # parse and get axes
    inputs_axes = parse_subscripts_part(inputs_part, True)
    outputs_axes = parse_subscripts_part(outputs_part, False)

    input_name_axes_dict: Dict[str, List[Axis]] = dict()
    output_name_axes_dict: Dict[str, List[Axis]] = dict()
    for input_axes in inputs_axes:
        for axis in input_axes:
            if axis.name in input_name_axes_dict:
                axes_of_name = input_name_axes_dict[axis.name]
                axes_of_name.append(axis)
            else:
                input_name_axes_dict[axis.name] = [axis]
    for output_axes in outputs_axes:
        for axis in output_axes:
            if axis.name in output_name_axes_dict:
                axes_of_name = output_name_axes_dict[axis.name]
                axes_of_name.append(axis)
            else:
                output_name_axes_dict[axis.name] = [axis]

    # create relation
    relation = AxisAxisRelation(inputs_axes, outputs_axes)
    input_names = list(input_name_axes_dict.keys())
    output_names = list(output_name_axes_dict.keys())
    for name in input_names:
        if name in output_names:
            input_axes = input_name_axes_dict[name]
            output_axes = output_name_axes_dict[name]
            # add equality relation
            relation.insert(
                *input_axes, *output_axes, relationship=AxisAxisRelation.RelationShip.EQ
            )
            # add dependent equality relation
            for i_axis in input_axes:
                for o_axis in output_axes:
                    relation.insert(
                        i_axis,
                        o_axis,
                        relationship=AxisAxisRelation.RelationShip.DEPEND_EQ,
                    )

    for i_name in input_names:
        for o_name in output_names:
            input_axes = input_name_axes_dict[i_name]
            output_axes = output_name_axes_dict[o_name]
            # add dependent split relation
            if len(i_name) > len(o_name) and o_name in i_name:
                for i_axis in input_axes:
                    for o_axis in output_axes:
                        relation.insert(
                            i_axis,
                            o_axis,
                            relationship=AxisAxisRelation.RelationShip.DEPEND_SPLIT,
                        )
            # add dependent merging relation
            elif len(o_name) > len(i_name) and i_name in o_name:
                for i_axis in input_axes:
                    for o_axis in output_axes:
                        relation.insert(
                            i_axis,
                            o_axis,
                            relationship=AxisAxisRelation.RelationShip.DEPEND_MERGE,
                        )

    return relation


class EinsumPrimitive(ABC):
    def __init__(self, inputs: List[FakeData], einsum_str: str) -> None:
        self.inputs = inputs
        self.einsum_str = einsum_str
        self.axes_relations = parse_einsum_str(einsum_str)

        self.dim_axis_relations = DimAxisRelation()

        # TODO: use relation instead of script string
        self.inputs_scripts, self.outputs_scripts = self.parse_einsum_str(
            self.einsum_str
        )
        # update scripts
        for scripts in (*self.inputs_scripts, *self.outputs_scripts):
            if scripts[0] == "_" and len(scripts) == 1:
                scripts.clear()
        # iter scripts
        _ = []
        for sp in (*self.inputs_scripts, *self.outputs_scripts):
            _ += sp
        self.iteration_scripts = set(_)
        # outputs
        self.outputs = self.run()

    def _run(self):
        dtype = self.inputs[0].dtype
        outputs_axes = self.axes_relations.outputs_axes
        rets = []
        for output_axes in outputs_axes:
            ret = FakeTensor(
                symbolic_shapes=dims((axis.name for axis in output_axes)),
                dtype=dtype,
                trace=self,
            )
            rets.append(ret)
        return rets

    def run(self):
        """
        check inputs and einsum, generate symbolic outputs
        """
        # TODO: dtypes of outputs
        dtype = self.inputs[0].dtype
        ret = []
        for output_scripts in self.outputs_scripts:
            ret.append(
                FakeTensor(
                    symbolic_shapes=dims(output_scripts), dtype=dtype, trace=self
                )
            )
        return ret

    def accept(self, visitor) -> None:
        """
        Accept a visitor for the visitor pattern.
        This method should be implemented by subclasses.
        """
        cls_name = self.__class__.__name__
        mth = getattr(visitor, f"visit_{cls_name}", None)
        if mth is None:
            raise RuntimeError("Expected visitor has function:  `{}`".format(cls_name))
        return mth(self)

    def parse_einsum_str(
        self,
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

        def parse_subscripts_part(part: str):
            # "ab,bc" -> ['ab', 'bc']
            part_subscripts = part.split(",")
            # 确保每个操作数都有下标，0维张量必须用'_'而不是空字符串
            for i, subscripts in enumerate(part_subscripts):
                if not subscripts:
                    raise ValueError(
                        f"第{i+1}个操作数的下标不能为空，0维张量请使用'_'表示"
                    )

            # ['ab', 'bc'] -> [['a', 'b'], ['b', 'c']]
            rets = []
            for part_subscript in part_subscripts:
                part_subscript_list = list(part_subscript)
                for idx in part_subscript_list:
                    if idx == "_" and len(part_subscript_list) != 1:
                        raise ValueError(f"错误的输入{part_subscript}")
                rets.append(part_subscript_list)
            return rets

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


class MapPrimitive(EinsumPrimitive):
    def __init__(
        self,
        lhs: FakeData,
        rhs: FakeData,
        einsum_str: str,
        op: ComputeOperator,
    ) -> None:
        super().__init__([lhs, rhs], einsum_str)

        # init scripts
        assert len(self.inputs_scripts) == 2
        self.lhs_scripts, self.rhs_scripts = self.inputs_scripts
        self.lhs = lhs
        self.rhs = rhs
        self.op = op
        self.output = self.outputs[0]
        self.output_scripts = self.outputs_scripts[0]


class ReducePrimitive(EinsumPrimitive):

    def __init__(
        self,
        x: FakeData,
        einsum_str: str,
        dims_to_reduce: Union[str, Sequence[str]],
        op: ComputeOperator,
    ) -> None:
        super().__init__([x], einsum_str)
        assert len(self.inputs_scripts) == 1
        self.x_scripts = self.inputs_scripts[0]  # only one input

        self.dims_to_reduce = (
            [dims_to_reduce]
            if isinstance(dims_to_reduce, str)
            else list(dims_to_reduce)
        )

        # 自己组合出ReduceFu
        self.op = op
        self.output = self.outputs[0]
        self.output_scripts = self.outputs_scripts[0]


class UnaryPrimitive(EinsumPrimitive):

    def __init__(self, x: FakeData, einsum_str: str, op: ComputeOperator):
        super().__init__(inputs=[x], einsum_str=einsum_str)
        self.x = x
        assert len(self.inputs_scripts) == 1
        self.x_scripts = self.inputs_scripts[0]  # only one input
        self.op = op
        self.output = self.outputs[0]
        self.output_scripts = self.outputs_scripts[0]


class CascadePrimitive(EinsumPrimitive):
    def __init__(
        self,
        inputs: Sequence[FakeData],
        graph,
        einsum_str: str,
    ):
        from Aipiler.graph import EinsumGraph

        super().__init__(list(inputs), einsum_str)
        self.graph: EinsumGraph = graph


class RearrangePrimitive(EinsumPrimitive):
    def __init__(self, inputs, einsum_str, **axes_length):
        super().__init__(inputs, einsum_str)
        self.input = self.inputs[0]
        self.inputs_script = self.inputs_scripts[0]
        self.outputs_script = self.outputs_scripts[0]
        self.axes_length: Dict[str, int] = axes_length

    def parse_einsum_str(self, equation: str):
        def parse_subscripts_part(part: str):
            # "ab,bc" -> ['ab', 'bc']
            part_subscripts = part.split(",")
            # 确保每个操作数都有下标，0维张量必须用'_'而不是空字符串
            for i, subscripts in enumerate(part_subscripts):
                if not subscripts:
                    raise ValueError(
                        f"第{i+1}个操作数的下标不能为空，0维张量请使用'_'表示"
                    )

            # ['ab', 'bc'] -> [['a', 'b'], ['b', 'c']]
            rets = []
            for part_subscript in part_subscripts:
                part_subscript_list = list(part_subscript)
                for idx in part_subscript_list:
                    if idx == "_" and len(part_subscript_list) != 1:
                        raise ValueError(f"错误的输入{part_subscript}")
                rets.append(part_subscript_list)
            return rets

        # 第一步：字符合法性检查
        # 创建包含所有允许字符的集合：小写字母a-z、下划线、逗号、箭头符号
        # TODO: support affine expression
        equation = equation.replace(" ", "")
        allowed_chars: Set[str] = set("abcdefghijklmnopqrstuvwxyz()->")

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
        input_ = parse_subscripts_part(input_part)
        output_ = parse_subscripts_part(output_part)

        # 第十步：验证输出下标合法性
        # 确保每个输出下标在输入中出现过（除了特殊的'_'标记）
        subscript_counts = []
        for input_subscript in input_:
            subscript_counts += input_subscript

        for subscript_list in output_:
            for sp in subscript_list:
                if sp != "_" and sp not in subscript_counts:
                    # 如果输出中有未在输入中出现过的下标，报错
                    raise ValueError(f"输出下标 '{sp}' 未在任何输入操作数中出现")
        return input_, output_


class PopulatePrimitive(EinsumPrimitive):

    def __init__(self):
        super().__init__(inputs=[], einsum_str="")
        pass


class EinsumBuilder:
    """
    A builder for creating Einsum primitives.
    This class is used to create Einsum primitives like Map, Reduce, Populate, and Unary.
    """

    @staticmethod
    def map(
        lhs: FakeData,
        rhs: FakeData,
        einsum_str: str,
        op: ComputeOperator,
    ) -> FakeData:
        assert lhs.dtype == rhs.dtype
        m = MapPrimitive(lhs, rhs, einsum_str, op)
        return m.output

    @staticmethod
    def reduce(
        x: FakeData,
        einsum_str: str,
        dim_to_reduce: Union[str, Sequence[str]],
        op: ComputeOperator,
    ) -> FakeData:
        return ReducePrimitive(x, einsum_str, dim_to_reduce, op).output

    @staticmethod
    def unary(x: FakeData, einsum_str: str, op: ComputeOperator) -> FakeData:
        return UnaryPrimitive(x, einsum_str, op).output

    @staticmethod
    def cascade(*args, subgraph, einsum_str):
        return CascadePrimitive(args, subgraph, einsum_str).outputs

    @staticmethod
    def populate() -> FakeData:
        pass
