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
from Aipiler.utils.printer import P


class DimAxisRelation:
    """维度和轴之间的双向映射关系管理器"""

    def __init__(self):
        self._dim_to_axis: Dict[Dim, Axis] = dict()
        self._axis_to_dim: Dict[Axis, Dim] = dict()

    @property
    def dims(self) -> List[Dim]:
        """获取所有维度"""
        return list(self._dim_to_axis.keys())

    @property
    def axes(self) -> List[Axis]:
        """获取所有轴"""
        return list(self._axis_to_dim.keys())

    def insert(self, dim: Dim, axis: Axis) -> None:
        """插入维度-轴映射关系

        Args:
            dim: 维度对象
            axis: 轴对象

        Raises:
            ValueError: 当维度或轴已存在时
        """
        if dim in self._dim_to_axis or axis in self._axis_to_dim:
            raise ValueError(f"维度-轴对 {dim} - {axis} 已存在")

        self._dim_to_axis[dim] = axis
        self._axis_to_dim[axis] = dim

    def get_axis(self, dim: Dim) -> Optional[Axis]:
        """根据维度获取轴"""
        return self._dim_to_axis.get(dim)

    def get_dim(self, axis: Axis) -> Optional[Dim]:
        """根据轴获取维度"""
        return self._axis_to_dim.get(axis)

    def __getitem__(self, key) -> Union[Dim, Axis]:
        """支持键访问，自动判断类型"""
        if isinstance(key, Dim) and key in self._dim_to_axis:
            return self._dim_to_axis[key]
        elif isinstance(key, Axis) and key in self._axis_to_dim:
            return self._axis_to_dim[key]
        else:
            raise KeyError(f"未找到键: {key}")

    def __repr__(self) -> str:
        """格式化输出维度-轴映射"""
        with P.section("维度到轴的映射"):
            with P.table(separator=" -> ", aligns=["c", "c"], col_widths=[30, 80]) as t:
                t.add_row("维度", "轴")
                t.add_row("-" * 10, "-" * 10)
                for dim, axis in self._dim_to_axis.items():
                    t.add_row(str(dim), str(axis))
            P.add_line()
        return str(P)


class RelationshipType(Enum):
    """轴关系类型枚举"""

    EQ = "equality_between_inputs_or_outputs"
    DEPEND_EQ = "dependency_and_equality_between_input_and_output"
    DEPEND_SPLIT = "split_one_input_axis_into_multi_output_axes"
    DEPEND_MERGE = "merge_multi_input_axes_into_one_output_axes"


class RelationshipManager:
    """轴关系管理器的基类"""

    def __init__(self):
        self.relationships: Dict[RelationshipType, Dict] = {
            RelationshipType.EQ: {},
            RelationshipType.DEPEND_EQ: {},
            RelationshipType.DEPEND_SPLIT: {},
            RelationshipType.DEPEND_MERGE: {},
        }

    def _get_relationship_dict(self, rel_type: RelationshipType) -> Dict:
        """获取指定类型的关系字典"""
        return self.relationships[rel_type]


class EqualityRelationshipHandler(RelationshipManager):
    """等价关系处理器"""

    def __init__(self):
        super().__init__()
        self.equal_dict: Dict[Axis, Set[Axis]] = {}

    def add_equality(self, *axes) -> None:
        """添加等价关系"""
        old_set: Optional[Set[Axis]] = None

        # 查找是否已存在包含这些轴的等价集合
        for axis in axes:
            if axis in self.equal_dict:
                old_set = self.equal_dict[axis]
                break

        # 创建或更新等价集合
        if old_set is None:
            old_set = set(axes)
        else:
            old_set.update(axes)

        # 更新所有轴的等价关系
        for axis in old_set:
            self.equal_dict[axis] = old_set


class DependencyRelationshipHandler(RelationshipManager):
    """依赖关系处理器"""

    def __init__(self):
        super().__init__()
        # 依赖等价关系
        self.depend_eq_input_to_output: Dict[Axis, Set[Axis]] = {}
        self.depend_eq_output_to_input: Dict[Axis, Set[Axis]] = {}

        # 依赖拆分关系
        self.depend_split_input_to_outputs: Dict[Axis, Set[Axis]] = {}
        self.depend_split_output_to_input: Dict[Axis, Axis] = {}

        # 依赖合并关系
        self.depend_merge_input_to_output: Dict[Axis, Axis] = {}
        self.depend_merge_output_to_inputs: Dict[Axis, Set[Axis]] = {}

    def add_dependency_equality(self, input_axis: Axis, output_axis: Axis) -> None:
        """添加依赖等价关系"""
        # 更新输入到输出的映射
        if input_axis in self.depend_eq_input_to_output:
            self.depend_eq_input_to_output[input_axis].add(output_axis)
        else:
            self.depend_eq_input_to_output[input_axis] = {output_axis}

        # 更新输出到输入的映射
        if output_axis in self.depend_eq_output_to_input:
            self.depend_eq_output_to_input[output_axis].add(input_axis)
        else:
            self.depend_eq_output_to_input[output_axis] = {input_axis}

    def add_inputAxis_split_to_outputAxes(
        self, input_axis: Axis, output_axes: List[Axis]
    ) -> None:
        """添加输入轴拆分到多个输出轴的关系"""
        for output_axis in output_axes:
            self.add_inputAxis_split_to_outputAxis(input_axis, output_axis)

    def add_inputAxis_split_to_outputAxis(
        self, input_axis: Axis, output_axis: Axis
    ) -> None:
        """添加依赖拆分关系"""
        # 更新输入到输出的映射
        if input_axis in self.depend_split_input_to_outputs:
            self.depend_split_input_to_outputs[input_axis].add(output_axis)
        else:
            self.depend_split_input_to_outputs[input_axis] = {output_axis}

        # 检查输出轴是否已有其他输入轴
        if (
            output_axis in self.depend_split_output_to_input
            and self.depend_split_output_to_input[output_axis] != input_axis
        ):
            raise ValueError(f"输出轴 {output_axis} 已与其他输入轴存在拆分关系")

        self.depend_split_output_to_input[output_axis] = input_axis

    def add_inputAxes_merge_to_outputAxis(
        self, input_axes: List[Axis], output_axis: Axis
    ) -> None:
        """添加输入轴拆分到多个输出轴的关系"""
        for input_axis in input_axes:
            self.add_inputAxis_merge_to_outputAxis(input_axis, output_axis)

    def add_inputAxis_merge_to_outputAxis(
        self, input_axis: Axis, output_axis: Axis
    ) -> None:
        """添加依赖合并关系"""
        # 检查输入轴是否已有其他输出轴
        if (
            input_axis in self.depend_merge_input_to_output
            and self.depend_merge_input_to_output[input_axis] != output_axis
        ):
            raise ValueError(f"输入轴 {input_axis} 已与其他输出轴存在合并关系")

        self.depend_merge_input_to_output[input_axis] = output_axis

        # 更新输出到输入的映射
        if output_axis in self.depend_merge_output_to_inputs:
            self.depend_merge_output_to_inputs[output_axis].add(input_axis)
        else:
            self.depend_merge_output_to_inputs[output_axis] = {input_axis}


class AxisAxisRelation:
    """轴与轴之间的关系管理器"""

    def __init__(self, inputs_axes: List[List[Axis]], outputs_axes: List[List[Axis]]):
        self._inputs_axes = inputs_axes
        self._outputs_axes = outputs_axes
        self.all_input_axes = [axis for axes_list in inputs_axes for axis in axes_list]
        self.all_output_axes = [
            axis for axes_list in outputs_axes for axis in axes_list
        ]

        # 初始化关系处理器
        self.equality_handler = EqualityRelationshipHandler()
        self.dependency_handler = DependencyRelationshipHandler()

    @property
    def inputs_axes(self) -> List[List[Axis]]:
        return self._inputs_axes

    @property
    def outputs_axes(self) -> List[List[Axis]]:
        return self._outputs_axes

    @overload
    def insert(
        self, input_axis: Axis, output_axis: Axis, relationship: RelationshipType
    ) -> None:
        """插入特定关系的轴对"""
        ...

    @overload
    def insert(
        self, *axes, relationship: RelationshipType = RelationshipType.EQ
    ) -> None:
        """插入等价关系的多个轴"""
        ...

    def insert(self, *args, **kwargs) -> None:
        """插入轴关系的统一接口"""
        # 解析参数
        relationship = self._parse_insert_arguments(*args, **kwargs)
        axes = args[:-1] if isinstance(args[-1], RelationshipType) else args

        if relationship == RelationshipType.EQ:
            self.equality_handler.add_equality(*axes)
        elif relationship == RelationshipType.DEPEND_EQ:
            if len(axes) != 2:
                raise ValueError("依赖等价关系需要恰好两个轴")
            self.dependency_handler.add_dependency_equality(axes[0], axes[1])
        elif relationship == RelationshipType.DEPEND_SPLIT:
            input_axis = axes[0]
            output_axes = axes[1:]
            self.dependency_handler.add_inputAxis_split_to_outputAxes(
                input_axis, output_axes
            )
        elif relationship == RelationshipType.DEPEND_MERGE:
            output_axis = axes[-1]
            input_axes = axes[0:-1]
            self.dependency_handler.add_inputAxes_merge_to_outputAxis(
                input_axes, output_axis
            )
        else:
            raise ValueError(f"不支持的关系类型: {relationship}")

    def _parse_insert_arguments(self, *args, **kwargs) -> RelationshipType:
        """解析insert方法的参数"""
        if "relationship" in kwargs:
            relationship = kwargs["relationship"]
        elif len(args) > 0 and isinstance(args[-1], RelationshipType):
            relationship = args[-1]
        else:
            relationship = RelationshipType.EQ

        if not isinstance(relationship, RelationshipType):
            raise ValueError("关系类型必须是RelationshipType枚举")

        return relationship

    def __repr__(self) -> str:
        """格式化输出所有轴关系"""
        sections = []

        # 等价关系
        if self.equality_handler.equal_dict:
            sections.append(self._format_equality_section())

        # 依赖等价关系
        if self.dependency_handler.depend_eq_input_to_output:
            sections.append(self._format_dependency_eq_section())

        # 依赖拆分关系
        if self.dependency_handler.depend_split_input_to_outputs:
            sections.append(self._format_dependency_split_section())

        # 依赖合并关系
        if self.dependency_handler.depend_merge_input_to_output:
            sections.append(self._format_dependency_merge_section())

        return "\n".join(sections)

    def _format_equality_section(self) -> str:
        """格式化等价关系部分"""
        with P.section("等价关系"):
            with P.table(separator=" | ", aligns=["c", "c"], col_widths=[30, 80]) as t:
                t.add_row("轴", "等价于")
                t.add_row("-" * 30, "-" * 80)
                for axis, equal_set in self.equality_handler.equal_dict.items():
                    t.add_row(str(axis), str(equal_set))
        return str(P)

    def _format_dependency_eq_section(self) -> str:
        """格式化依赖等价关系部分"""
        with P.section("依赖等价关系"):
            with P.table(separator=" | ", aligns=["c", "c"], col_widths=[30, 80]) as t:
                t.add_row("输入轴", "依赖等价于")
                t.add_row("-" * 30, "-" * 80)
                for (
                    input_axis,
                    output_set,
                ) in self.dependency_handler.depend_eq_input_to_output.items():
                    t.add_row(str(input_axis), str(output_set))
        return str(P)

    def _format_dependency_split_section(self) -> str:
        """格式化依赖拆分关系部分"""
        with P.section("依赖拆分关系"):
            with P.table(separator=" | ", aligns=["c", "c"], col_widths=[30, 80]) as t:
                t.add_row("输入轴", "拆分为")
                t.add_row("-" * 30, "-" * 80)
                for (
                    input_axis,
                    output_set,
                ) in self.dependency_handler.depend_split_input_to_outputs.items():
                    t.add_row(str(input_axis), str(output_set))
        return str(P)

    def _format_dependency_merge_section(self) -> str:
        """格式化依赖合并关系部分"""
        with P.section("依赖合并关系"):
            with P.table(separator=" | ", aligns=["c", "c"], col_widths=[30, 80]) as t:
                t.add_row("输入轴", "合并为")
                t.add_row("-" * 30, "-" * 80)
                for (
                    input_axis,
                    output_axis,
                ) in self.dependency_handler.depend_merge_input_to_output.items():
                    t.add_row(str(input_axis), str(output_axis))
        return str(P)


class EinsumPrimitive(ABC):
    def __init__(self, inputs: List[FakeData], einsum_str: str) -> None:
        self.inputs = inputs
        self.einsum_str = einsum_str
        self.inputs_axes, self.outputs_axes = self._parse_einsum_str(einsum_str)
        # relations between axes
        self.axes_relations: AxisAxisRelation = self._bind_axes(
            self.inputs_axes, self.outputs_axes
        )
        # instance outputs
        self.outputs = self._run()
        # relations between dim and axis
        self.dim_axis_relations = self._bind_dim_axis()
        # iter space
        self.iteration_axes = self._get_iterspace()

    def _parse_einsum_str(self, equation: str):
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
                Axis(name, self, is_input, idx_inside, idx)
                for idx_inside, name in enumerate(names)
            ]
            return result

        def parse_subscripts_part(part: str, is_input: bool):
            # "ab,bc" -> ['ab', 'bc']
            part_subscripts = part.split(",")
            # 确保每个操作数都有下标，0维张量必须用'_'而不是空字符串
            for i, subscripts in enumerate(part_subscripts):
                if not subscripts:
                    raise ValueError(
                        f"第{i+1}个操作数的下标不能为空，0维张量请使用'_'表示"
                    )
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
        return inputs_axes, outputs_axes

    def _bind_axes(self, inputs_axes: List[List[Axis]], outputs_axes: List[List[Axis]]):
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
                    *input_axes,
                    *output_axes,
                    relationship=RelationshipType.EQ,
                )
                # add dependent equality relation
                for i_axis in input_axes:
                    for o_axis in output_axes:
                        relation.insert(
                            i_axis,
                            o_axis,
                            relationship=RelationshipType.DEPEND_EQ,
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
                                relationship=RelationshipType.DEPEND_SPLIT,
                            )
                # add dependent merging relation
                elif len(o_name) > len(i_name) and i_name in o_name:
                    for i_axis in input_axes:
                        for o_axis in output_axes:
                            relation.insert(
                                i_axis,
                                o_axis,
                                relationship=RelationshipType.DEPEND_MERGE,
                            )

        return relation

    def _bind_dim_axis(self):
        dim_axis_relations = DimAxisRelation()
        for axes, io_tensor in zip(
            (*self.inputs_axes, *self.outputs_axes), (*self.inputs, *self.outputs)
        ):
            if isinstance(io_tensor, FakeScalar):
                continue
            io_tensor: FakeTensor
            for axis, dim in zip(axes, io_tensor.symbolic_shapes):
                dim_axis_relations.insert(dim, axis)
        return dim_axis_relations

    def _get_iterspace(self) -> Set[Axis]:
        """Give an example of iterspace
        a(bc) -> abc ==> iterspace: Axis("a"), Axis("b"), Axis("c")
        Returns:
            Set[Axis]: example axes to iterate
        """
        valide_axes = [
            axis
            for axis in (
                *self.axes_relations.all_input_axes,
                *self.axes_relations.all_output_axes,
            )
            if not axis.is_combined and not axis.is_scalar
        ]
        iter_axis_name = set()
        iter_space = set()
        for axis in valide_axes:
            if axis.name in iter_axis_name:
                continue
            else:
                iter_axis_name.add(axis.name)
                iter_space.add(axis)
        return iter_space

    def _run(self):
        dtype = self.inputs[0].dtype
        outputs_axes = self.axes_relations.outputs_axes
        rets: List[FakeTensor] = []
        for output_axes in outputs_axes:
            axes_names = [axis.name for axis in output_axes]
            output = FakeTensor(
                symbolic_shapes=dims(axes_names),
                dtype=dtype,
                trace=self,
            )
            rets.append(output)

        return rets

    def __repr__(self):
        from Aipiler.utils.namer import N

        with P.section(
            "{}: {}".format(N.get_or_create_name_of(self), self.__class__.__name__)
        ):
            with P.table() as t:
                t.add_row("inputs", self.inputs)
                t.add_row("outputs", self.outputs)
                t.add_row("einsum_str", self.einsum_str)
            P.add_line(str(self.dim_axis_relations))
            P.add_line(str(self.axes_relations))
        return str(P)


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
        assert len(self.inputs_axes) == 2
        self.lhs_axes, self.rhs_axes = self.inputs_axes
        self.lhs = lhs
        self.rhs = rhs
        self.op = op
        self.output = self.outputs[0]
        self.output_axes = self.outputs_axes[0]


class ReducePrimitive(EinsumPrimitive):

    def __init__(
        self,
        x: FakeData,
        einsum_str: str,
        dims_to_reduce: Union[str, Sequence[str]],
        op: ComputeOperator,
    ) -> None:
        super().__init__([x], einsum_str)
        assert len(self.inputs) == 1
        self.x = x
        self.x_axes = self.inputs_axes[0]  # only one input

        self.dims_to_reduce = (
            [dims_to_reduce]
            if isinstance(dims_to_reduce, str)
            else list(dims_to_reduce)
        )
        self.op = op
        self.output = self.outputs[0]
        self.output_axes = self.outputs_axes[0]


class UnaryPrimitive(EinsumPrimitive):

    def __init__(self, x: FakeData, einsum_str: str, op: ComputeOperator):
        super().__init__([x], einsum_str)
        self.x = x
        assert len(self.inputs) == 1
        self.x_axes = self.inputs_axes[0]  # only one input
        self.op = op
        self.output = self.outputs[0]
        self.output_axes = self.inputs_axes[0]


class RearrangePrimitive(EinsumPrimitive):
    def __init__(
        self, inputs: List[FakeTensor] | FakeTensor, einsum_str: str, **axes_length
    ):
        if isinstance(inputs, FakeTensor):
            inputs = [inputs]
        # initialize
        self.inputs = inputs
        self.einsum_str = einsum_str
        # wild means no related tensor
        (
            self.inputs_axes,
            self.outputs_axes,
            self.wild_input_axis,
            self.wild_input_dim,
        ) = self._parse_einsum_str(einsum_str)
        # relations between axes
        self.axes_relations: AxisAxisRelation = self._bind_axes(
            self.inputs_axes, self.outputs_axes, self.wild_input_axis
        )
        # instance outputs
        self.outputs = self._run()
        # relations between dim and axis
        self.dim_axis_relations = self._bind_dim_axis()
        self.dim_axis_relations.insert(self.wild_input_dim, self.wild_input_axis)
        # iter space
        self.iteration_axes = self._get_iterspace()

        self.input = self.inputs[0]
        self.input_axes = self.inputs_axes[0]
        self.output = self.outputs[0]
        self.output_axes = self.outputs_axes[0]
        self.sized_output_dims: Dict[Dim, int] = self._handle_sized_output_dims(
            axes_length
        )

    def _parse_einsum_str(self, equation: str):
        inputs_axes, outputs_axes = super()._parse_einsum_str(equation)
        wild_axis: Axis = None
        wild_dim: Dim = None
        if len(self.inputs) != 1:
            assert len(inputs_axes) == 1
            input_axes = inputs_axes[0]
            wild_axis = input_axes.pop(0)
            wild_dim_len = len(self.inputs)
            wild_dim = Dim(wild_dim_len)
            for _ in range(1, wild_dim_len):
                new_axes = []
                for axis in input_axes:
                    new_a = Axis(
                        axis.name,
                        axis._from_prim,
                        axis._is_from_input,
                        axis._idx_in_script,
                        _,
                    )
                    new_axes.append(new_a)
                inputs_axes.append(new_axes)
            assert len(input_axes) == len(self.inputs)
        return inputs_axes, outputs_axes, wild_axis, wild_dim

    def _bind_axes(
        self,
        inputs_axes: List[List[Axis]],
        outputs_axes: List[List[Axis]],
        wild_input_axis: Axis,
    ):
        if wild_input_axis is None:
            return super()._bind_axes(inputs_axes, outputs_axes)
        else:
            assert isinstance(wild_input_axis, Axis)
            return super()._bind_axes([[wild_input_axis], *inputs_axes], outputs_axes)

    def _handle_sized_output_dims(self, axes_length: Dict[str, int]):
        sized_dims: Dict[Dim, int] = dict()
        for d in self.output.symbolic_shapes:
            if d.size in axes_length:
                size = axes_length[d.size]
                d._set_size(size)
                sized_dims[d] = size
        return sized_dims

    def __repr__(self):
        from Aipiler.utils.namer import N

        with P.section(
            "{}: {}".format(N.get_or_create_name_of(self), self.__class__.__name__)
        ):
            with P.table(separator=" | ") as t:
                t.add_row("input", self.input)
                t.add_row("output", self.output)
                t.add_row("einsum_str", self.einsum_str)
                t.add_row("wild axis", self.wild_input_axis)
                t.add_row("wild dim", self.wild_input_dim)
                t.add_row("sized dims", self.sized_output_dims)
                t.add_row("iter space", self.iteration_axes)
            P.add_line(str(self.dim_axis_relations))
            P.add_line(str(self.axes_relations))
        return str(P)


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
    def cascade(*args, subgraph, einsum_str) -> FakeData:
        return CascadePrimitive(args, subgraph, einsum_str).outputs

    @staticmethod
    def rearrange(
        inputs: Union[FakeTensor, List[FakeTensor]],
        einsum_str: str,
        **axes_length: Dict[str, int],
    ) -> FakeTensor:
        if isinstance(inputs, FakeData):
            inputs = [inputs]
        if not all(isinstance(i, FakeTensor) for i in inputs):
            raise ValueError(
                "Rearrage only expects FakeTensor as input, got [{}]".format(
                    ", ".join(type(i) for i in inputs)
                )
            )
        rearrange_primitive = RearrangePrimitive(inputs, einsum_str, **axes_length)
        return rearrange_primitive.output

    @staticmethod
    def populate() -> FakeData:
        pass
