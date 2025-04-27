class MemoryHierarchy:
    """内存层次结构描述"""

    def __init__(self):
        self.levels = {
            "DRAM": {"size": 8 * 1024 * 1024 * 1024, "latency": 100},
            "L3_Cache": {"size": 8 * 1024 * 1024, "latency": 40},
            "L2_Cache": {"size": 256 * 1024, "latency": 10},
            "L1_Cache": {"size": 32 * 1024, "latency": 3},
            "Register": {"size": 128, "latency": 1},
        }


class DataMovement:
    """数据移动操作"""

    def __init__(self, source_view, target_view):
        self.source = source_view
        self.target = target_view
        self.transfer_size = calculate_size(source_view)
        self.is_sync = True  # 同步或异步
