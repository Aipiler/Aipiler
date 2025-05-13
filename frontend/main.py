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

# from .converter import PyTorchGraphConverter


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
    example_args = (torch.randn(3, 256, 256),)
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
            for arg in node.args:
                print(f"    - {arg}, type: {type(arg)}")

            if hasattr(node, "meta") and "val" in node.meta:
                output_fakeTensor = node.meta["val"]

                print(f"  输出Tensor: {output_fakeTensor}")
                print(f"  输出形状: {output_fakeTensor.size()}")
        elif node.op == "output":
            print(f"  参数: {node.args}")
            if hasattr(node, "meta") and "val" in node.meta:
                print(f"  输出形状: {node.meta['val']}")

    # # 转换为自定义图结构
    # print("\nStep 3: 转换为自定义图结构")
    # converter = PyTorchGraphConverter()
    # custom_graph = converter.convert(exported_program)

    # # # 打印自定义图信息
    # print("\nStep 4: 转换后的自定义图信息")
    # print("-" * 50)
    # print(f"图名称: {custom_graph.name}")
    # print(f"节点数量: {len(custom_graph.nodes)}")
    # print(f"边数量: {len(custom_graph.edges)}")

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
