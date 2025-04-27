from abc import ABC, abstractmethod


# 执行策略 - 处理不同内存层次的执行方式
class ExecutionStrategy:
    """执行策略基类 - 关注如何在特定内存层次执行操作"""

    def __init__(self, memory_level):
        self.memory_level = memory_level

    @abstractmethod
    def apply(self, compute_op, *data_edges):
        """应用特定内存层次的执行策略"""
        pass


class TensorExecutionStrategy(ExecutionStrategy):
    """张量级执行策略"""

    def __init__(self):
        super().__init__("DRAM")

    def apply(self, compute_op, *data_edges):
        """在张量级别执行计算"""
        # 获取张量数据
        tensor_views = [edge.views["tensor"] for edge in data_edges]
        # 调用计算操作的execute方法
        return compute_op.execute(*tensor_views)


class TileExecutionStrategy(ExecutionStrategy):
    """数据块级执行策略"""

    def __init__(self, tile_shape, loop_order=None):
        super().__init__("L2_Cache")
        self.tile_shape = tile_shape
        self.loop_order = loop_order or []

    def apply(self, compute_op, *data_edges):
        """在数据块级别执行计算"""
        # 获取分块后的数据
        tile_views = []
        for edge in data_edges:
            if not edge.views["tile"]:
                # 如果还没有分块，则创建分块
                edge.create_tile_views([self.tile_shape])
            tile_views.append(edge.views["tile"])

        # 创建分块执行循环
        results = []
        for tiles in zip(*tile_views):
            # 调用同样的计算操作，但输入是分块后的数据
            result_tile = compute_op.execute(*tiles)
            results.append(result_tile)

        return results


class VectorExecutionStrategy(ExecutionStrategy):
    """向量级执行策略"""

    def __init__(self, vector_length):
        super().__init__("Register")
        self.vector_length = vector_length

    def apply(self, compute_op, *data_edges):
        """在向量级别执行计算"""
        # 类似的向量化执行逻辑
        # ...
