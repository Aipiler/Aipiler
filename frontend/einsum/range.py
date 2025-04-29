from typing import List, Optional, Tuple, Dict, Any, Set, Callable, Union


class Range:
    """
    Represents a half-open interval [lower_bound, upper_bound).
    A valid range requires upper_bound > lower_bound, except when upper_bound = 0,
    which is a special case allowing upper_bound to be less than or equal to lower_bound.
    """

    # 定义无穷大常量
    INFINITY = float("inf")

    def __init__(self, lower_bound: int = 0, upper_bound: int = 0):
        # 边界检查 - 对于有效区间，上界必须大于下界（左闭右开区间）
        if upper_bound != 0 and upper_bound <= lower_bound:
            raise ValueError(
                f"Upper bound {upper_bound} must be > lower bound {lower_bound} for a valid range"
            )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def is_empty(self) -> bool:
        """Check if this range is empty. A range is empty if lower_bound >= upper_bound."""
        return self.lower_bound >= self.upper_bound

    def contains(self, value: int) -> bool:
        """Check if value is contained in this range. Value must be >= lower_bound and < upper_bound."""
        return self.lower_bound <= value < self.upper_bound

    def intersection(self, other: "Range") -> "Range":
        """Return the intersection of this range with another range."""
        lower = max(self.lower_bound, other.lower_bound)
        upper = min(self.upper_bound, other.upper_bound)
        if lower >= upper:
            # 返回空区间
            return Range(0, 0)
        return Range(lower, upper)

    def union(self, other: "Range") -> Optional["Range"]:
        """
        Return the union of this range with another range if they overlap or are adjacent.
        Returns None if ranges are disjoint and not adjacent.
        """
        # 检查是否为空区间
        if self.is_empty() and other.is_empty():
            return Range(0, 0)  # 两个空区间的并集是空区间
        elif self.is_empty():
            return Range(
                other.lower_bound, other.upper_bound
            )  # 如果当前区间为空，返回另一个区间
        elif other.is_empty():
            return Range(
                self.lower_bound, self.upper_bound
            )  # 如果另一个区间为空，返回当前区间

        # 检查是否有重叠或相邻
        if (self.upper_bound < other.lower_bound - 1) or (
            other.upper_bound < self.lower_bound - 1
        ):
            return None

        lower = min(self.lower_bound, other.lower_bound)
        upper = max(self.upper_bound, other.upper_bound)
        return Range(lower, upper)

    def __eq__(self, other):
        if not isinstance(other, Range):
            return False
        return (
            self.lower_bound == other.lower_bound
            and self.upper_bound == other.upper_bound
        )

    def __repr__(self):
        if self.upper_bound == self.INFINITY:
            return f"[{self.lower_bound}, ∞)"
        return f"[{self.lower_bound}, {self.upper_bound})"


class CompoundRange:
    """表示可能不连续的复合区间集合"""

    def __init__(self, *initial_ranges: Range):
        """初始化复合区间，可选择传入一个或多个初始区间"""
        self.ranges: List[Range] = []  # 存储多个Range对象

        # 添加初始区间（如果有）
        for range_obj in initial_ranges:
            self.add_range(range_obj)

    def add_range(self, range_obj: Range):
        """添加一个区间，并自动合并相邻或重叠的区间"""
        if range_obj.is_empty():
            return

        # 尝试合并
        merged = False
        for i, existing_range in enumerate(self.ranges):
            union_range = existing_range.union(range_obj)
            if union_range:
                self.ranges[i] = union_range
                merged = True
                break

        if not merged:
            self.ranges.append(range_obj)

        # 合并后再次检查重叠
        self._consolidate()

    def _consolidate(self):
        """合并所有可合并的区间"""
        if not self.ranges:
            return

        # 排序区间
        self.ranges.sort(key=lambda r: (r.lower_bound, r.upper_bound))

        i = 0
        while i < len(self.ranges) - 1:
            curr = self.ranges[i]
            next_range = self.ranges[i + 1]
            union_range = curr.union(next_range)

            if union_range:
                self.ranges[i] = union_range
                self.ranges.pop(i + 1)
            else:
                i += 1

    def intersection(self, other: "CompoundRange") -> "CompoundRange":
        """计算与另一个复合区间的交集"""
        result = CompoundRange()
        for r1 in self.ranges:
            for r2 in other.ranges:
                intersection = r1.intersection(r2)
                if not intersection.is_empty():
                    result.add_range(intersection)
        return result

    def is_empty(self) -> bool:
        """检查复合区间是否为空"""
        return len(self.ranges) == 0 or all(r.is_empty() for r in self.ranges)

    def contains(self, value: int) -> bool:
        """检查值是否在复合区间中"""
        return any(r.contains(value) for r in self.ranges)

    def __eq__(self, other) -> bool:
        """比较两个CompoundRange对象是否相等"""
        if not isinstance(other, CompoundRange):
            return False

        # 首先比较ranges列表长度
        if len(self.ranges) != len(other.ranges):
            return False

        # 确保在比较前两者都被排序和合并
        self._consolidate()
        other._consolidate()

        # 逐个比较每个范围区间
        for i in range(len(self.ranges)):
            if self.ranges[i] != other.ranges[i]:
                return False

        return True

    def __repr__(self):
        if not self.ranges:
            return "∅"
        return " ∪ ".join([str(r) for r in self.ranges])


# 完整测试 - 修改后以适应左闭右开区间的规则
import unittest


class TestRange(unittest.TestCase):
    """Range类的单元测试"""

    def test_initialization(self):
        """测试Range的初始化"""
        # 正常初始化
        r = Range(1, 5)
        self.assertEqual(r.lower_bound, 1)
        self.assertEqual(r.upper_bound, 5)

        # 默认初始化
        r = Range()
        self.assertEqual(r.lower_bound, 0)
        self.assertEqual(r.upper_bound, 0)

        # 上界为0的特殊情况
        r = Range(10, 0)
        self.assertEqual(r.lower_bound, 10)
        self.assertEqual(r.upper_bound, 0)

        # 无穷大作为上界
        r = Range(10, Range.INFINITY)
        self.assertEqual(r.lower_bound, 10)
        self.assertEqual(r.upper_bound, Range.INFINITY)

        # 测试错误情况 - 上界不大于下界
        with self.assertRaises(ValueError):
            Range(10, 10)  # 上界等于下界

        with self.assertRaises(ValueError):
            Range(10, 5)  # 上界小于下界

    def test_is_empty(self):
        """测试is_empty方法"""
        # 空区间 - 默认构造
        self.assertTrue(Range().is_empty())

        # 空区间 - 特殊情况 upper_bound = 0
        self.assertTrue(Range(5, 0).is_empty())

        # 非空区间
        self.assertFalse(Range(1, 5).is_empty())
        self.assertFalse(Range(0, 1).is_empty())
        self.assertFalse(Range(-10, 10).is_empty())

    def test_contains(self):
        """测试contains方法"""
        r = Range(1, 5)

        # 包含的情况
        self.assertTrue(r.contains(1))
        self.assertTrue(r.contains(2))
        self.assertTrue(r.contains(4))

        # 不包含的情况
        self.assertFalse(r.contains(0))
        self.assertFalse(r.contains(5))  # 上界是开区间
        self.assertFalse(r.contains(10))

        # 无穷大情况
        r_inf = Range(10, Range.INFINITY)
        self.assertTrue(r_inf.contains(10))
        self.assertTrue(r_inf.contains(1000000))
        self.assertFalse(r_inf.contains(9))

    def test_intersection(self):
        """测试intersection方法"""
        # 相交区间
        r1 = Range(1, 5)
        r2 = Range(3, 7)
        intersection = r1.intersection(r2)
        self.assertEqual(intersection, Range(3, 5))

        # 包含关系
        r1 = Range(1, 10)
        r2 = Range(3, 7)
        intersection = r1.intersection(r2)
        self.assertEqual(intersection, Range(3, 7))

        # 不相交区间
        r1 = Range(1, 3)
        r2 = Range(5, 7)
        intersection = r1.intersection(r2)
        self.assertTrue(intersection.is_empty())

        # 边界相交情况
        r1 = Range(1, 5)
        r2 = Range(5, 10)
        intersection = r1.intersection(r2)
        self.assertTrue(intersection.is_empty())

        # 空区间交集
        r1 = Range(1, 5)
        r2 = Range()
        intersection = r1.intersection(r2)
        self.assertTrue(intersection.is_empty())

    def test_union(self):
        """测试union方法"""
        # 相交区间
        r1 = Range(1, 5)
        r2 = Range(3, 7)
        union = r1.union(r2)
        self.assertEqual(union, Range(1, 7))

        # 相邻区间
        r1 = Range(1, 5)
        r2 = Range(5, 10)
        union = r1.union(r2)
        self.assertEqual(union, Range(1, 10))

        # 几乎相邻区间（差1）
        r1 = Range(1, 5)
        r2 = Range(6, 10)
        union = r1.union(r2)
        self.assertEqual(union, Range(1, 10))

        # 不相交不相邻区间
        r1 = Range(1, 3)
        r2 = Range(5, 7)
        union = r1.union(r2)
        self.assertIsNone(union)

        # 包含关系
        r1 = Range(1, 10)
        r2 = Range(3, 7)
        union = r1.union(r2)
        self.assertEqual(union, Range(1, 10))

        # 空区间合并
        r1 = Range(1, 5)
        r2 = Range()
        union = r1.union(r2)
        self.assertEqual(union, Range(1, 5))  # 空区间与非空区间合并应返回非空区间

        # 两个空区间合并
        r1 = Range()
        r2 = Range()
        union = r1.union(r2)
        self.assertEqual(union, Range(0, 0))  # 两个空区间合并应返回空区间

    def test_equality(self):
        """测试相等性判断"""
        # 相等区间
        self.assertEqual(Range(1, 5), Range(1, 5))
        self.assertEqual(Range(), Range())
        self.assertEqual(Range(10, Range.INFINITY), Range(10, Range.INFINITY))

        # 不相等区间
        self.assertNotEqual(Range(1, 5), Range(1, 6))
        self.assertNotEqual(Range(2, 5), Range(1, 5))
        self.assertNotEqual(Range(), Range(0, 1))

        # 与其他类型比较
        self.assertNotEqual(Range(1, 5), "not a range")
        self.assertNotEqual(Range(1, 5), (1, 5))
        self.assertNotEqual(Range(1, 5), None)

    def test_representation(self):
        """测试字符串表示"""
        self.assertEqual(str(Range(1, 5)), "[1, 5)")
        self.assertEqual(str(Range()), "[0, 0)")
        self.assertEqual(str(Range(10, Range.INFINITY)), "[10, ∞)")


class TestCompoundRange(unittest.TestCase):
    """CompoundRange类的单元测试"""

    def test_initialization(self):
        """测试CompoundRange的初始化"""
        # 空初始化
        cr = CompoundRange()
        self.assertEqual(len(cr.ranges), 0)

        # 单个区间初始化
        cr = CompoundRange(Range(1, 5))
        self.assertEqual(len(cr.ranges), 1)
        self.assertEqual(cr.ranges[0], Range(1, 5))

        # 多个区间初始化
        cr = CompoundRange(Range(1, 5), Range(7, 10))
        self.assertEqual(len(cr.ranges), 2)
        self.assertEqual(cr.ranges[0], Range(1, 5))
        self.assertEqual(cr.ranges[1], Range(7, 10))

        # 空区间被过滤
        cr = CompoundRange(Range(), Range(1, 5))
        self.assertEqual(len(cr.ranges), 1)
        self.assertEqual(cr.ranges[0], Range(1, 5))

        # 重叠区间会合并
        cr = CompoundRange(Range(1, 5), Range(3, 7))
        self.assertEqual(len(cr.ranges), 1)
        self.assertEqual(cr.ranges[0], Range(1, 7))

        # 相邻区间会合并
        cr = CompoundRange(Range(1, 5), Range(5, 10))
        self.assertEqual(len(cr.ranges), 1)
        self.assertEqual(cr.ranges[0], Range(1, 10))

    def test_add_range(self):
        """测试add_range方法"""
        cr = CompoundRange()

        # 添加单个区间
        cr.add_range(Range(1, 5))
        self.assertEqual(len(cr.ranges), 1)
        self.assertEqual(cr.ranges[0], Range(1, 5))

        # 添加不重叠的区间
        cr.add_range(Range(7, 10))
        self.assertEqual(len(cr.ranges), 2)
        self.assertEqual(cr.ranges[0], Range(1, 5))
        self.assertEqual(cr.ranges[1], Range(7, 10))

        # 添加重叠区间
        cr.add_range(Range(4, 8))
        self.assertEqual(len(cr.ranges), 1)
        self.assertEqual(cr.ranges[0], Range(1, 10))

        # 添加空区间
        cr.add_range(Range())
        self.assertEqual(len(cr.ranges), 1)  # 不变

        # 添加可合并的多个区间
        cr = CompoundRange()
        cr.add_range(Range(1, 5))
        cr.add_range(Range(7, 10))
        cr.add_range(Range(5, 7))  # 应该将前两个区间连接起来
        self.assertEqual(len(cr.ranges), 1)
        self.assertEqual(cr.ranges[0], Range(1, 10))

        # 添加新的不重叠区间
        cr.add_range(Range(15, 20))
        self.assertEqual(len(cr.ranges), 2)
        self.assertEqual(cr.ranges[0], Range(1, 10))
        self.assertEqual(cr.ranges[1], Range(15, 20))

        # 测试多次添加和合并的复杂情况
        cr = CompoundRange()
        cr.add_range(Range(1, 5))
        cr.add_range(Range(10, 15))
        cr.add_range(Range(20, 25))
        cr.add_range(Range(4, 11))  # 应该合并前两个区间
        self.assertEqual(len(cr.ranges), 2)
        self.assertEqual(cr.ranges[0], Range(1, 15))
        self.assertEqual(cr.ranges[1], Range(20, 25))

        cr.add_range(Range(15, 20))  # 应该合并所有区间
        self.assertEqual(len(cr.ranges), 1)
        self.assertEqual(cr.ranges[0], Range(1, 25))

    def test_consolidate(self):
        """测试_consolidate方法"""
        # 通过添加需要合并的区间间接测试
        cr = CompoundRange()
        cr.add_range(Range(5, 10))
        cr.add_range(Range(1, 3))
        cr.add_range(Range(3, 6))

        # 经过合并后，应该有 [1, 10)
        self.assertEqual(len(cr.ranges), 1)
        self.assertEqual(cr.ranges[0], Range(1, 10))

        # 测试排序功能
        cr = CompoundRange()
        cr.add_range(Range(5, 10))
        cr.add_range(Range(1, 3))

        # 确保排序正确
        self.assertEqual(cr.ranges[0], Range(1, 3))
        self.assertEqual(cr.ranges[1], Range(5, 10))

    def test_intersection(self):
        """测试intersection方法"""
        # 相交复合区间
        cr1 = CompoundRange(Range(1, 5), Range(10, 15))
        cr2 = CompoundRange(Range(3, 7), Range(9, 12))
        intersection = cr1.intersection(cr2)
        self.assertEqual(len(intersection.ranges), 2)
        self.assertEqual(intersection.ranges[0], Range(3, 5))
        self.assertEqual(intersection.ranges[1], Range(10, 12))

        # 无交集的复合区间
        cr1 = CompoundRange(Range(1, 3), Range(5, 7))
        cr2 = CompoundRange(Range(4, 5), Range(8, 10))
        intersection = cr1.intersection(cr2)
        self.assertTrue(intersection.is_empty())

        # 部分交集
        cr1 = CompoundRange(Range(1, 10))
        cr2 = CompoundRange(Range(5, 15))
        intersection = cr1.intersection(cr2)
        self.assertEqual(len(intersection.ranges), 1)
        self.assertEqual(intersection.ranges[0], Range(5, 10))

        # 包含关系
        cr1 = CompoundRange(Range(1, 20))
        cr2 = CompoundRange(Range(5, 10), Range(12, 15))
        intersection = cr1.intersection(cr2)
        self.assertEqual(len(intersection.ranges), 2)
        self.assertEqual(intersection.ranges[0], Range(5, 10))
        self.assertEqual(intersection.ranges[1], Range(12, 15))

        # 空区间交集
        cr1 = CompoundRange(Range(1, 5))
        cr2 = CompoundRange()
        intersection = cr1.intersection(cr2)
        self.assertTrue(intersection.is_empty())

    def test_is_empty(self):
        """测试is_empty方法"""
        # 空复合区间
        self.assertTrue(CompoundRange().is_empty())
        self.assertTrue(CompoundRange(Range()).is_empty())

        # 含特殊空区间
        cr = CompoundRange()
        cr.add_range(Range())
        cr.add_range(Range(5, 0))
        self.assertTrue(cr.is_empty())

        # 非空复合区间
        self.assertFalse(CompoundRange(Range(1, 5)).is_empty())
        self.assertFalse(CompoundRange(Range(1, 5), Range(7, 10)).is_empty())

        # 混合情况
        cr = CompoundRange(Range(), Range(1, 5))
        self.assertFalse(cr.is_empty())

    def test_contains(self):
        """测试contains方法"""
        cr = CompoundRange(Range(1, 5), Range(10, 15))

        # 包含在第一个区间
        self.assertTrue(cr.contains(1))
        self.assertTrue(cr.contains(4))

        # 包含在第二个区间
        self.assertTrue(cr.contains(10))
        self.assertTrue(cr.contains(14))

        # 不包含
        self.assertFalse(cr.contains(0))
        self.assertFalse(cr.contains(5))
        self.assertFalse(cr.contains(7))
        self.assertFalse(cr.contains(15))
        self.assertFalse(cr.contains(20))

        # 空复合区间
        cr_empty = CompoundRange()
        self.assertFalse(cr_empty.contains(0))
        self.assertFalse(cr_empty.contains(10))

    def test_equality(self):
        """测试相等性判断"""
        # 测试 1: 空区间比较
        cr1 = CompoundRange()
        cr2 = CompoundRange()
        self.assertEqual(cr1, cr2)

        # 测试 2: 单个相同区间
        cr1 = CompoundRange(Range(1, 5))
        cr2 = CompoundRange(Range(1, 5))
        self.assertEqual(cr1, cr2)

        # 测试 3: 不同区间
        cr1 = CompoundRange(Range(1, 5))
        cr2 = CompoundRange(Range(2, 6))
        self.assertNotEqual(cr1, cr2)

        # 测试 4: 不同数量的区间
        cr1 = CompoundRange(Range(1, 5), Range(7, 10))
        cr2 = CompoundRange(Range(1, 5))
        self.assertNotEqual(cr1, cr2)

        # 测试 5: 相同的多个区间
        cr1 = CompoundRange(Range(1, 5), Range(7, 10))
        cr2 = CompoundRange(Range(1, 5), Range(7, 10))
        self.assertEqual(cr1, cr2)

        # 测试 6: 不同顺序添加但应该相等的区间
        cr1 = CompoundRange(Range(1, 5), Range(7, 10))
        cr2 = CompoundRange(Range(7, 10), Range(1, 5))
        self.assertEqual(cr1, cr2)

        # 测试 7: 有重叠需要合并的区间
        cr1 = CompoundRange(Range(1, 5), Range(4, 10))
        cr2 = CompoundRange(Range(1, 10))
        self.assertEqual(cr1, cr2)

        # 测试 8: 相邻区间的合并
        cr1 = CompoundRange(Range(1, 5), Range(5, 10))
        cr2 = CompoundRange(Range(1, 10))
        self.assertEqual(cr1, cr2)

        # 测试 9: 与其他类型比较
        cr1 = CompoundRange(Range(1, 5))
        self.assertNotEqual(cr1, "not a CompoundRange")
        self.assertNotEqual(cr1, Range(1, 5))

    def test_representation(self):
        """测试字符串表示"""
        self.assertEqual(str(CompoundRange()), "∅")
        self.assertEqual(str(CompoundRange(Range(1, 5))), "[1, 5)")
        self.assertEqual(
            str(CompoundRange(Range(1, 5), Range(7, 10))), "[1, 5) ∪ [7, 10)"
        )

        # 测试合并后的表示
        cr = CompoundRange(Range(1, 5), Range(4, 10))
        self.assertEqual(str(cr), "[1, 10)")


if __name__ == "__main__":
    unittest.main()
