import torch
import torch.nn as nn
from torch.export import export
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 导入自定义实现
from node import Node, PlaceholderNode, ComputeNode, OutputNode
from edge import Edge
from graph import Graph
from compute import Compute, EinsumExpression
from data import Data
from .opMapper import einsum_mapper


class PyTorchGraphConverter:
    """将PyTorch导出的图转换为自定义图结构"""

    def __init__(self):
        self.graph = Graph("PyTorchConverted")
        self.op_mapper = einsum_mapper
        self.logger = logging.getLogger(self.__class__.__name__)

    def convert(self, exported_program: torch.export.ExportedProgram) -> Graph:
        """转换PyTorch导出的程序到自定义图结构"""
        self.logger.info("开始转换PyTorch导出的程序")
        torch_graph = exported_program.graph

        # 第一步：创建所有节点
        self._create_nodes(torch_graph)

        # 第二步：创建所有边
        self._create_edges(torch_graph)

        self.logger.info(
            f"转换完成：创建了{len(self.graph.nodes)}个节点和{len(self.graph.edges)}条边"
        )
        return self.graph

    def _create_nodes(self, torch_graph: torch.fx.Graph) -> None:
        """从PyTorch图创建所有节点"""
        self.logger.info("创建节点...")

        for node in torch_graph.nodes:
            if node.op == "placeholder":
                # 创建输入占位符节点
                custom_node = PlaceholderNode(name=f"{node.name}")
                self.logger.debug(f"创建占位符节点: {custom_node.name}")

                # 如果有形状信息，设置它
                if hasattr(node, "meta") and "val" in node.meta:
                    # TODO: 根据FakeTensor的信息设置output data信息。
                    pass
                    # tensor_shape = node.meta["val"].shape
                    # custom_node.set_shape(tensor_shape)
                    # self.logger.debug(f"  设置形状: {tensor_shape}")

            elif node.op == "output":
                # 创建输出节点
                custom_node = OutputNode(name=f"{node.name}")
                self.logger.debug(f"创建输出节点: {custom_node.name}")

            elif node.op == "get_attr":
                pass

            else:
                # 创建计算节点
                handler = self.op_mapper.get_handler(node.target)
                einsum_expr = handler(node)
                custom_node = ComputeNode(einsum=einsum, name=f"{node.name}")
                self.logger.debug(f"创建计算节点: {custom_node.name}")

            # 添加节点到图中并保存映射
            node_id = self.graph.add_node(custom_node)
            self.node_map[node.name] = node_id

    def _create_einsum_for_node(self, node: torch.fx.Node) -> EinsumExpression:
        """为PyTorch节点创建对应的Compute实例"""

        # 仅针对call_function类型node，创建专门的Compute对象
        if node.op != "call_function":
            raise ValueError(
                f"Unsupported node operation: {node.op}. Only 'call_function' is supported."
            )

        # 如果没有匹配到特定操作，创建通用的Compute对象
        return EinsumExpression(node.target)

    def _create_edges(self, torch_graph) -> None:
        """从PyTorch图创建所有边"""
        self.logger.info("创建边...")

        # 遍历每个节点及其输入参数
        for node in torch_graph.nodes:
            if node.op == "output":
                # 连接到输出节点的边
                if (
                    hasattr(node, "args")
                    and len(node.args) > 0
                    and isinstance(node.args[0], (list, tuple))
                ):
                    for i, arg in enumerate(node.args[0]):
                        if hasattr(arg, "name"):  # 检查是否是节点引用
                            source_id = self.node_map[arg.name]
                            target_id = self.node_map[node.name]
                            edge_id = self.graph.connect(source_id, target_id)
                            self.logger.debug(
                                f"创建边: {arg.name} -> {node.name} (ID: {edge_id})"
                            )
            else:
                # 对于其它节点，检查其输入参数
                target_id = self.node_map[node.name]

                # 处理位置参数
                if hasattr(node, "args"):
                    for arg in node.args:
                        if hasattr(arg, "name"):  # 检查是否是节点引用
                            source_id = self.node_map[arg.name]
                            edge_id = self.graph.connect(source_id, target_id)
                            self.logger.debug(
                                f"创建边: {arg.name} -> {node.name} (ID: {edge_id})"
                            )

                # 处理关键字参数
                if hasattr(node, "kwargs"):
                    for key, val in node.kwargs.items():
                        if hasattr(val, "name"):  # 检查是否是节点引用
                            source_id = self.node_map[val.name]
                            edge_id = self.graph.connect(source_id, target_id)
                            self.logger.debug(
                                f"创建边: {val.name} -> {node.name} (键: {key}, ID: {edge_id})"
                            )
