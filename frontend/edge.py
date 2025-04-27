from data import Data
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING
import uuid

# 避免循环导入问题
if TYPE_CHECKING:
    from node import Node, PlaceholderNode, OutputNode


class Edge:
    """计算图中的数据边"""

    def __init__(
        self,
        source_node: "Node",
        target_node: "Node",
        data: Optional[Data] = None,
        name: Optional[str] = None,
    ):
        self.id = str(uuid.uuid4())  # 添加唯一ID
        self.source = source_node
        self.target = target_node
        self.data = data
        self.name = (
            name if name is not None else f"{source_node.name} -> {target_node.name}"
        )
        self.status = "created"  # 边的状态: "created", "active", "stale"

        # 类型验证：检查源节点是否为OutputNode，检查目标节点是否为PlaceholderNode
        from node import PlaceholderNode, OutputNode

        if isinstance(source_node, OutputNode):
            raise TypeError(
                f"OutputNode '{source_node.name}' cannot be used as a source node"
            )
        if isinstance(target_node, PlaceholderNode):
            raise TypeError(
                f"PlaceholderNode '{target_node.name}' cannot be used as a target node"
            )

        try:
            # 将边添加到节点的输入/输出列表中
            source_node.add_output_edge(self)
            target_node.add_input_edge(self)
        except TypeError as e:
            # 如果添加失败，确保不会留下部分连接的边
            self.disconnect()
            raise e

    def disconnect(self) -> None:
        """从源节点和目标节点中移除此边的连接"""
        try:
            if self.source is not None:
                if self in self.source.output_edges:
                    self.source.remove_output_edge(self)
        except TypeError:
            pass  # 忽略因特殊节点类型导致的异常

        try:
            if self.target is not None:
                if self in self.target.input_edges:
                    self.target.remove_input_edge(self)
        except TypeError:
            pass  # 忽略因特殊节点类型导致的异常

        self.status = "disconnected"

    def get_data(self) -> Optional[Data]:
        """获取边上的数据"""
        return self.data

    def set_data(self, data: Optional[Data]) -> None:
        """设置边上的数据并更新状态"""
        prev_data = self.data
        self.data = data

        if data is not None:
            self.status = "active"
        elif prev_data is not None and data is None:
            self.status = "stale"

    def clear_data(self) -> None:
        """清除边上的数据"""
        self.data = None
        self.status = "stale"

    def get_shape(self) -> Optional[Any]:
        """获取边上数据的形状信息（如果可用）"""
        if self.data is None:
            return None

        # 如果数据对象有shape属性，则返回shape
        if hasattr(self.data, "shape"):
            return self.data.shape

        # 或者尝试从源节点获取形状信息
        if hasattr(self.source, "get_shape"):
            return self.source.get_shape()
        elif hasattr(self.source, "output_shape"):
            return self.source.output_shape

        return None

    def is_active(self) -> bool:
        """检查边是否处于活动状态（有数据）"""
        return self.data is not None

    def __str__(self):
        data_str = str(self.data) if self.data is not None else "None"
        if len(data_str) > 30:
            data_str = data_str[:27] + "..."
        return f"Edge({self.name}, status={self.status}, data={data_str})"

    def __repr__(self):
        return self.__str__()
