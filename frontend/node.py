from compute import EinsumExpression
from data import Data
from typing import List, Dict, Any, Optional, Set, Tuple
import uuid
from abc import ABC, abstractmethod


class Node(ABC):
    """统一的计算节点抽象类"""

    def __init__(
        self, einsum: Optional[EinsumExpression] = None, name: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())  # 添加唯一ID
        self.name = name if name is not None else f"Node-{self.id[:8]}"
        self.einsum = einsum
        self.input_edges = []  # 输入边列表
        self.output_edges = []  # 输出边列表
        self.status = "initialized"  # 节点状态: "initialized", "ready", "running", "completed", "error"

    def add_input_edge(self, edge):
        """添加输入边"""
        if edge not in self.input_edges:
            self.input_edges.append(edge)

    def add_output_edge(self, edge):
        """添加输出边"""
        if edge not in self.output_edges:
            self.output_edges.append(edge)

    def remove_input_edge(self, edge):
        """移除输入边"""
        if edge in self.input_edges:
            self.input_edges.remove(edge)

    def remove_output_edge(self, edge):
        """移除输出边"""
        if edge in self.output_edges:
            self.output_edges.remove(edge)

    def get_input_data(self) -> List[Data]:
        """获取所有输入数据"""
        return [edge.data for edge in self.input_edges if edge.data is not None]

    def get_input_nodes(self) -> List["Node"]:
        """获取所有输入节点"""
        return [edge.source for edge in self.input_edges]

    def get_output_nodes(self) -> List["Node"]:
        """获取所有输出节点"""
        return [edge.target for edge in self.output_edges]

    def propagate_result(self, result: Optional[Data]) -> None:
        """将结果传播到所有输出边"""
        for edge in self.output_edges:
            edge.data = result

    def is_ready(self) -> bool:
        """检查节点是否准备好执行（所有输入边都有数据）"""
        return all(edge.data is not None for edge in self.input_edges)

    def set_status(self, status: str) -> None:
        """设置节点状态"""
        self.status = status

    def get_status(self) -> str:
        """获取节点状态"""
        return self.status

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self):
        return self.__str__()


class PlaceholderNode(Node):
    """占位符节点，表示输入数据"""

    def __init__(self, data: Optional[Data] = None, name: Optional[str] = None):
        super().__init__(name=name)
        self.data = data
        self.shape = None  # 存储数据形状

    def set_data(self, data: Data) -> None:
        """设置占位符节点的数据"""
        self.data = data
        self.set_status("ready")

    def get_data(self) -> Optional[Data]:
        """获取占位符节点的数据"""
        return self.data

    def set_shape(self, shape: Tuple) -> None:
        """设置数据形状信息"""
        self.shape = shape

    def get_shape(self) -> Optional[Tuple]:
        """获取数据形状信息"""
        return self.shape

    def add_input_edge(self, edge):
        """占位符节点不能添加输入边"""
        raise TypeError(
            f"PlaceholderNode '{self.name}' cannot add input edges. It is a source node."
        )

    def remove_input_edge(self, edge):
        """占位符节点不能移除输入边，因为它不应该有任何输入边"""
        raise TypeError(
            f"PlaceholderNode '{self.name}' cannot remove input edges. It is a source node."
        )


class ComputeNode(Node):
    """计算节点，表示一个计算操作"""

    def __init__(self, einsum: EinsumExpression, name: Optional[str] = None):
        if einsum is None:
            raise ValueError("ComputeNode must have a non-null compute instance")
        super().__init__(einsum=einsum, name=name)
        self.output_shape = None  # 存储推断的输出数据形状

    def verify(self) -> Tuple[bool, Optional[str]]:
        """验证计算节点的输入是否符合要求

        返回:
            Tuple[bool, Optional[str]]: (验证是否通过, 错误信息)
        """
        if self.einsum is None:
            return False, "没有设置计算操作"

        pass

    def infer_return_shape(self) -> Optional[Tuple]:
        """推断计算节点输出数据的形状

        返回:
            Optional[Tuple]: 推断的输出数据形状
        """
        pass


class OutputNode(Node):
    """输出节点，表示计算图的输出"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        self.result = None
        self.output_shape = None

    def get_result(self) -> Optional[Data]:
        """获取输出节点的结果"""
        input_data = self.get_input_data()
        if input_data:
            self.result = input_data[0]  # 通常输出节点只有一个输入
        return self.result

    def infer_output_shape(self) -> Optional[Tuple]:
        """推断输出形状，基于输入节点"""
        for edge in self.input_edges:
            source = edge.source
            if hasattr(source, "get_shape"):
                self.output_shape = source.get_shape()
            elif hasattr(source, "output_shape"):
                self.output_shape = source.output_shape
        return self.output_shape

    def is_valid(self) -> bool:
        """检查输出节点是否有效（至少有一个输入边）"""
        return len(self.input_edges) > 0

    def add_output_edge(self, edge):
        """输出节点不能添加输出边"""
        raise TypeError(
            f"OutputNode '{self.name}' cannot add output edges. It is a sink node."
        )

    def remove_output_edge(self, edge):
        """输出节点不能移除输出边，因为它不应该有任何输出边"""
        raise TypeError(
            f"OutputNode '{self.name}' cannot remove output edges. It is a sink node."
        )
