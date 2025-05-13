from compute import Compute
from data import Data
from typing import List, Dict, Any, Optional, Set, Tuple, Union, Type
import uuid
from collections import deque
import logging
from edge import Edge
from node import Node, PlaceholderNode, ComputeNode, OutputNode


class Graph:
    """节点管理器，负责管理计算图中的所有节点"""

    def __init__(self, name: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.name = name if name is not None else f"Graph-{self.id[:8]}"
        self.nodes = {}  # 存储所有节点，key为节点ID，value为节点对象
        self.edges = {}  # 存储所有边，key为边ID，value为边对象
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_node(self, node: Node) -> str:
        """添加节点到管理器

        Args:
            node: 要添加的节点

        Returns:
            str: 节点ID
        """
        # 使用节点的ID作为键，而不是名称
        if node.id in self.nodes:
            raise ValueError(f"Node with ID '{node.id}' already exists")

        self.nodes[node.id] = node
        return node.id

    def remove_node(self, node_id: str) -> bool:
        """从管理器中删除节点

        Args:
            node_id: 节点ID

        Returns:
            bool: 是否成功删除
        """
        if node_id not in self.nodes:
            self.logger.warning(f"Node '{node_id}' not found")
            return False

        node = self.nodes[node_id]

        # 获取与该节点相关的所有边
        edges_to_remove = []
        for edge_id, edge in self.edges.items():
            if edge.source == node or edge.target == node:
                edges_to_remove.append(edge_id)

        # 删除相关的边
        for edge_id in edges_to_remove:
            self.remove_edge(edge_id)

        # 删除节点
        del self.nodes[node_id]
        return True

    def get_node(self, node_id: str) -> Optional[Node]:
        """获取指定ID的节点

        Args:
            node_id: 节点ID

        Returns:
            Node: 找到的节点，如果不存在则返回None
        """
        return self.nodes.get(node_id)

    def get_node_by_name(self, name: str) -> Optional[Node]:
        """根据名称查找节点

        Args:
            name: 节点名称

        Returns:
            Node: 找到的节点，如果不存在则返回None
        """
        for node in self.nodes.values():
            if node.name == name:
                return node
        return None

    def get_all_nodes(self) -> List[Node]:
        """获取所有节点

        Returns:
            List[Node]: 所有节点的列表
        """
        return list(self.nodes.values())

    def connect(
        self, source_id: str, target_id: str, data: Optional[Data] = None
    ) -> Optional[str]:
        """连接两个节点

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            data: 初始数据（可选）

        Returns:
            str: 创建的边ID，如果失败则返回None
        """
        source_node = self.get_node(source_id)
        target_node = self.get_node(target_id)

        if source_node is None:
            self.logger.error(f"Source node '{source_id}' not found")
            return None

        if target_node is None:
            self.logger.error(f"Target node '{target_id}' not found")
            return None

        try:
            # 创建边（现在Edge构造函数会自动添加到节点）
            edge = Edge(source_node, target_node, data)

            # 保存边
            self.edges[edge.id] = edge
            return edge.id
        except TypeError as e:
            self.logger.error(f"Failed to connect nodes: {str(e)}")
            return None

    def disconnect(self, source_id: str, target_id: str) -> bool:
        """断开两个节点的连接

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID

        Returns:
            bool: 是否成功断开连接
        """
        source_node = self.get_node(source_id)
        target_node = self.get_node(target_id)

        if source_node is None or target_node is None:
            return False

        # 查找连接这两个节点的边
        edges_to_remove = []
        for edge_id, edge in self.edges.items():
            if edge.source == source_node and edge.target == target_node:
                edges_to_remove.append(edge_id)

        # 删除边
        success = False
        for edge_id in edges_to_remove:
            if self.remove_edge(edge_id):
                success = True

        return success

    def remove_edge(self, edge_id: str) -> bool:
        """删除边

        Args:
            edge_id: 边ID

        Returns:
            bool: 是否成功删除
        """
        if edge_id not in self.edges:
            return False

        edge = self.edges[edge_id]

        # 断开边的连接（Edge类中的方法会处理与节点的关系）
        edge.disconnect()

        # 删除边
        del self.edges[edge_id]

        return True

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """获取指定ID的边

        Args:
            edge_id: 边ID

        Returns:
            Edge: 找到的边，如果不存在则返回None
        """
        return self.edges.get(edge_id)

    def get_all_edges(self) -> List[Edge]:
        """获取所有边

        Returns:
            List[Edge]: 所有边的列表
        """
        return list(self.edges.values())

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """构建节点依赖图

        Returns:
            Dict[str, Set[str]]: 依赖图，key为节点ID，value为依赖该节点的节点ID集合
        """
        dependency_graph = {node_id: set() for node_id in self.nodes}

        for edge in self.edges.values():
            source_id = edge.source.id
            target_id = edge.target.id
            dependency_graph[target_id].add(source_id)

        return dependency_graph

    def _topological_sort(self) -> List[str]:
        """拓扑排序，确定节点执行顺序

        Returns:
            List[str]: 排序后的节点ID列表
        """
        dependency_graph = self._build_dependency_graph()

        # 查找入度为0的节点
        in_degree = {node_id: len(deps) for node_id, deps in dependency_graph.items()}
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])

        result = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            # 更新依赖于当前节点的节点的入度
            for other_id, deps in dependency_graph.items():
                if node_id in deps:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        # 检查是否有环
        if len(result) != len(self.nodes):
            self.logger.error("Cycle detected in the computation graph")
            return []

        return result

    def validate_graph(self) -> Tuple[bool, List[str]]:
        """验证计算图是否有效

        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []

        # 检查是否有环
        if not self._topological_sort():
            errors.append("Graph contains cycles")

        # 检查每个ComputeNode的输入是否有效
        for node in self.nodes.values():
            if isinstance(node, ComputeNode):
                is_valid, error_msg = node.verify()
                if not is_valid:
                    errors.append(f"Node '{node.name}' validation failed: {error_msg}")

        # 检查所有OutputNode是否有输入
        for node in self.nodes.values():
            if isinstance(node, OutputNode) and not node.is_valid():
                errors.append(f"Output node '{node.name}' has no inputs")

        return len(errors) == 0, errors

    def infer_shapes(self) -> Dict[str, Any]:
        """推断图中所有节点的输出形状

        Returns:
            Dict[str, Any]: 节点ID到形状的映射
        """
        # 按拓扑顺序推断形状
        shapes = {}
        execution_order = self._topological_sort()

        for node_id in execution_order:
            node = self.nodes[node_id]

            if isinstance(node, PlaceholderNode):
                shapes[node_id] = node.get_shape()
            elif isinstance(node, ComputeNode):
                shapes[node_id] = node.infer_return_shape()
            elif isinstance(node, OutputNode):
                shapes[node_id] = node.infer_output_shape()

        return shapes

    def execute(
        self, start_nodes: Optional[List[str]] = None
    ) -> Dict[str, Optional[Data]]:
        """执行计算图

        Args:
            start_nodes: 起始节点ID列表，如果为None则自动确定

        Returns:
            Dict[str, Data]: 每个节点的执行结果，key为节点ID，value为输出数据
        """
        # 由于去除了execute方法，我们需要修改这个实现
        self.logger.warning(
            "The execute method is deprecated as Node.execute has been removed"
        )
        return {}

    def get_placeholder_nodes(self) -> List[PlaceholderNode]:
        """获取所有占位符节点

        Returns:
            List[PlaceholderNode]: 所有占位符节点的列表
        """
        return [
            node for node in self.nodes.values() if isinstance(node, PlaceholderNode)
        ]

    def get_compute_nodes(self) -> List[ComputeNode]:
        """获取所有计算节点

        Returns:
            List[ComputeNode]: 所有计算节点的列表
        """
        return [node for node in self.nodes.values() if isinstance(node, ComputeNode)]

    def get_output_nodes(self) -> List[OutputNode]:
        """获取所有输出节点

        Returns:
            List[OutputNode]: 所有输出节点的列表
        """
        return [node for node in self.nodes.values() if isinstance(node, OutputNode)]

    def clear(self):
        """清空管理器中的所有节点和边"""
        # 先断开所有边的连接
        for edge in list(self.edges.values()):
            edge.disconnect()

        self.nodes.clear()
        self.edges.clear()

    def get_node_inputs(self, node_id: str) -> List[Tuple[str, Edge]]:
        """获取节点的所有输入边及其源节点

        Args:
            node_id: 节点ID

        Returns:
            List[Tuple[str, Edge]]: 源节点ID和边的元组列表
        """
        node = self.get_node(node_id)
        if node is None:
            return []

        return [(edge.source.id, edge) for edge in node.input_edges]

    def get_node_outputs(self, node_id: str) -> List[Tuple[str, Edge]]:
        """获取节点的所有输出边及其目标节点

        Args:
            node_id: 节点ID

        Returns:
            List[Tuple[str, Edge]]: 目标节点ID和边的元组列表
        """
        node = self.get_node(node_id)
        if node is None:
            return []

        return [(edge.target.id, edge) for edge in node.output_edges]

    def find_nodes_by_compute_type(
        self, compute_type: Type[Compute]
    ) -> List[ComputeNode]:
        """根据计算类型查找节点

        Args:
            compute_type: 计算类型

        Returns:
            List[ComputeNode]: 匹配的节点列表
        """
        return [
            node
            for node in self.nodes.values()
            if isinstance(node, ComputeNode)
            and node.compute is not None
            and isinstance(node.compute, compute_type)
        ]

    def to_dict(self) -> Dict:
        """将计算图转换为字典，方便序列化

        Returns:
            Dict: 表示计算图的字典
        """
        nodes_dict = {}
        for node_id, node in self.nodes.items():
            node_type = node.__class__.__name__
            nodes_dict[node_id] = {
                "id": node_id,
                "name": node.name,
                "type": node_type,
                "compute_type": (
                    type(node.compute).__name__
                    if hasattr(node, "compute") and node.compute
                    else None
                ),
                "status": node.status,
            }

        edges_list = []
        for edge_id, edge in self.edges.items():
            edges_list.append(
                {
                    "id": edge_id,
                    "source": edge.source.id,
                    "target": edge.target.id,
                    "name": edge.name,
                    "status": edge.status,
                }
            )

        return {
            "id": self.id,
            "name": self.name,
            "nodes": nodes_dict,
            "edges": edges_list,
        }

    def from_dict(self, data: Dict) -> None:
        """从字典中恢复计算图

        Args:
            data: 表示计算图的字典
        """
        # 此方法需要额外实现，需要知道如何根据类型创建节点
        # 这里只提供一个框架
        self.clear()

        self.id = data.get("id", str(uuid.uuid4()))
        self.name = data.get("name", f"Graph-{self.id[:8]}")

        # 需要先创建所有节点，然后再创建边
        self.logger.warning("from_dict method is not fully implemented")
