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


class PyTorchGraphConverter:
    """将PyTorch导出的图转换为自定义图结构"""

    def __init__(self):
        self.graph = Graph("PyTorchConverted")
        self.node_map = {}  # PyTorch节点名称到自定义节点ID的映射
        self.logger = logging.getLogger(self.__class__.__name__)
        self.aten_op_2_compute_mapping = {}

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
                einsum = self._create_einsum_for_node(node)
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


def main():
    # 创建一个简单的模型
    class M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1
            )
            self.relu = torch.nn.ReLU()
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            a = self.conv(x)
            return self.maxpool(self.relu(a))

    # 导出模型
    print("Step 1: 创建并导出PyTorch模型")
    example_args = (torch.randn(1, 3, 256, 256),)
    exported_program = export(M(), args=example_args)

    # 打印原始图信息
    print("\nStep 2: 原始PyTorch图信息")
    print("-" * 50)
    print(exported_program.graph)
    print("-" * 50)

    # 打印节点详情
    print("\nPyTorch图中的节点详情:")
    for node in exported_program.graph.nodes:
        print(f"Node: {node.name}, Op: {node.op}")
        if node.op == "placeholder":
            print(f"  参数: {node.args}")
            if hasattr(node, "meta") and "val" in node.meta:
                print(f"  输出形状: {node.meta['val']}")
        elif node.op == "call_function":
            print(f"  函数: {node.target}")
            print(f"  参数: {node.args}")
            if hasattr(node, "meta") and "val" in node.meta:
                print(f"  输出形状: {node.meta['val']}")
        elif node.op == "output":
            print(f"  参数: {node.args}")
            if hasattr(node, "meta") and "val" in node.meta:
                print(f"  输出形状: {node.meta['val']}")

    # # 转换为自定义图结构
    print("\nStep 3: 转换为自定义图结构")
    converter = PyTorchGraphConverter()
    custom_graph = converter.convert(exported_program)

    # # 打印自定义图信息
    print("\nStep 4: 转换后的自定义图信息")
    print("-" * 50)
    print(f"图名称: {custom_graph.name}")
    print(f"节点数量: {len(custom_graph.nodes)}")
    print(f"边数量: {len(custom_graph.edges)}")

    # # 打印节点信息
    # print("\n节点详情:")
    # for node_id, node in custom_graph.nodes.items():
    #     print(f"- {node.name} (ID: {node.id}, 类型: {node.__class__.__name__})")
    #     if hasattr(node, "compute") and node.compute:
    #         print(f"  计算操作: {node.compute.description}")
    #     if hasattr(node, "output_shape") and node.output_shape:
    #         print(f"  输出形状: {node.output_shape}")
    #     print(f"  输入边数: {len(node.input_edges)}")
    #     print(f"  输出边数: {len(node.output_edges)}")

    # # 打印边信息
    # print("\n边详情:")
    # for edge_id, edge in custom_graph.edges.items():
    #     print(f"- {edge} (ID: {edge.id})")
    #     print(f"  源节点: {edge.source.name}")
    #     print(f"  目标节点: {edge.target.name}")

    # # 验证图结构
    # print("\nStep 5: 验证图结构")
    # is_valid, errors = custom_graph.validate_graph()
    # print(f"图结构验证: {'通过' if is_valid else '失败'}")
    # if not is_valid:
    #     for error in errors:
    #         print(f"- {error}")

    # # 推断形状
    # print("\nStep 6: 形状推断")
    # shapes = custom_graph.infer_shapes()
    # print("形状推断结果:")
    # for node_id, shape in shapes.items():
    #     node = custom_graph.get_node(node_id)
    #     print(f"- {node.name}: {shape}")


if __name__ == "__main__":
    main()
