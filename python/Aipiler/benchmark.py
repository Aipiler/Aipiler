import os
import re
import statistics
import tempfile
import logging
import subprocess
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import namedtuple
from os import PathLike
import numpy as np
import iree.runtime as rt
from iree.runtime import VmModule
from iree.runtime.benchmark import (
    benchmark_module,
    benchmark_exe,
    BenchmarkTimeoutError,
    BenchmarkToolError,
)


__all__ = ["BenchmarkConfig", "BenchmarkResult", "BenchmarkRunner"]

DTYPE_TO_ABI_TYPE = {
    np.dtype(np.float32): "f32",
    np.dtype(np.int32): "i32",
    np.dtype(np.int64): "i64",
    np.dtype(np.float64): "f64",
    np.dtype(np.int16): "i16",
    np.dtype(np.float16): "f16",
    np.dtype(np.int8): "i8",
    np.dtype(np.bool_): "i1",
}


@dataclass
class BenchmarkConfig:
    """configuration for benchmark runs"""

    num_runs: int = 10
    timeout: Optional[float] = None
    entry_function: Optional[str] = None
    data_dir: str = "./benchmark_data"
    cleanup_after: bool = False
    verbose: bool = False
    min_runs_for_stats: int = 3
    benchmark_flags: Dict[str, Any] = field(default_factory=dict)
    statistical_analysis: bool = True


@dataclass
class BenchmarkResult:
    """container for benchmark results with statistical analysis"""

    # 基础信息
    benchmark_name: str
    config_used: BenchmarkConfig

    # 运行结果
    num_runs: int
    successful_runs: int
    failed_runs: int

    # 原始IREE结果
    raw_iree_results: List[rt.benchmark.BenchmarkResult]

    # 时间统计 (解析后的数值)
    time_values: List[float]
    cpu_time_values: List[float]

    # 统计指标
    mean_time: float = 0.0
    median_time: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    std_dev_time: Optional[float] = None
    cv_time: Optional[float] = None

    mean_cpu_time: float = 0.0
    median_cpu_time: float = 0.0
    min_cpu_time: float = 0.0
    max_cpu_time: float = 0.0
    std_dev_cpu_time: Optional[float] = None
    cv_cpu_time: Optional[float] = None

    # 性能指标
    throughput: float = 0.0
    cpu_throughput: float = 0.0
    stability_rating: str = "N/A"

    # 错误信息
    error_details: List[str] = field(default_factory=list)


class BenchmarkRunner:
    """benchmark runner with IREE compatibility and advanced features"""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        if self.config.verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def _get_benchmark_exe(self) -> str:
        """Get the path to the IREE benchmark executable"""
        return benchmark_exe()

    def _prepare_inputs_for_iree(self, inputs: List[Union[str]]) -> List[str]:
        """
        Prepare inputs in IREE benchmark format
        Compatible with official IREE benchmark_module input format
        """
        formatted_inputs = []

        for inp in inputs:
            if isinstance(inp, str):
                # 字符串输入，可能是文件路径或格式化字符串
                formatted_inputs.append(f"@{inp}")
                continue

            # numpy数组输入，报错，benchmark只能接收地址
            else:
                raise ValueError(f"Unsupported input type: {type(inp)}")

        return formatted_inputs

    def _save_inputs_to_files(self, inputs: List[np.ndarray]) -> List[str]:
        """Save numpy arrays to files and return file paths"""

        save_dir = self.config.data_dir
        os.makedirs(save_dir, exist_ok=True)

        file_paths = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, np.ndarray):
                file_path = os.path.join(save_dir, f"input_{i}.npy")
                np.save(file_path, inp)
                file_paths.append(os.path.abspath(file_path))

                if self.config.verbose:
                    self.logger.info(f"Saved input {i} to: {file_path}")

        return file_paths

    def _run_single_benchmark(
        self,
        module: Union[VmModule, PathLike],
        entry_function: str = None,
        inputs: List[str] = [],
        device: str = "local-task",
    ) -> list[rt.benchmark.BenchmarkResult]:
        """
        Run a single benchmark using IREE's benchmark tool
        """
        return benchmark_module(module, entry_function, inputs, device=device)

    def _parse_time_value(self, time_str: str) -> Optional[float]:
        """
        Parse time value from IREE benchmark output
        Handles various formats like "123.45 ms", "1.23 us", etc.
        """
        if not isinstance(time_str, str):
            return None

        # 匹配数字和单位的正则表达式
        pattern = r"(\d+\.?\d*)\s*(ns|us|ms|s)?"
        match = re.search(pattern, time_str.lower())

        if not match:
            return None

        value = float(match.group(1))
        unit = match.group(2) or "s"  # 默认为秒

        # 转换为秒
        unit_multipliers = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}

        return value * unit_multipliers.get(unit, 1.0)

    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical measures"""
        if not values:
            return {}

        stats = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

        if len(values) > 1:
            stats["std_dev"] = statistics.stdev(values)
            stats["cv"] = stats["std_dev"] / stats["mean"] if stats["mean"] > 0 else 0

        stats["throughput"] = 1.0 / stats["mean"] if stats["mean"] > 0 else 0
        return stats

    def _get_stability_rating(self, cv: Optional[float]) -> str:
        """Get stability rating based on coefficient of variation"""
        if cv is None:
            return "N/A"

        if cv < 0.05:
            return "优秀"
        elif cv < 0.1:
            return "良好"
        elif cv < 0.2:
            return "一般"
        else:
            return "较差"

    def run_benchmark(
        self,
        module: Union[VmModule, PathLike],
        entry_function: str = None,
        inputs: Union[List[np.ndarray], List[str]] = [],
        benchmark_name: str = "benchmark",
        device: str = "local-task",
    ) -> BenchmarkResult:
        """
        Run enhanced benchmark with statistical analysis

        Args:
            moduleIREE VmModule or path to module file:
            inputs: Input data (numpy arrays or formatted strings)
            benchmark_name: Name for this benchmark

        Returns:
            EnhancedBenchmarkResult with detailed statistics
        """
        if self.config.verbose:
            self.logger.info(f"开始基准测试: {benchmark_name}")
            self.logger.info(f"配置: {self.config.num_runs} 次运行")

        # 准备输入
        if inputs and isinstance(inputs[0], np.ndarray):
            # 保存文件
            saved_files = self._save_inputs_to_files(inputs)
            # 转换为IREE格式
            formatted_inputs = self._prepare_inputs_for_iree(saved_files)
        elif inputs and isinstance(inputs[0], str):
            formatted_inputs = inputs
            saved_files = []
        else:
            raise ValueError(f"Unsupported type: {type(inputs[0])}")

        # 运行基准测试
        raw_results = []
        successful_runs = 0
        error_details = []

        for i in range(self.config.num_runs):
            if self.config.verbose:
                self.logger.info(f"运行 {i+1}/{self.config.num_runs}, ...")

            try:
                result = self._run_single_benchmark(
                    module,
                    entry_function=entry_function,
                    inputs=formatted_inputs,
                    device=device,
                )[0]
                # print(f"result: {result}")
                raw_results.append(result)
                successful_runs += 1

                if self.config.verbose:
                    self.logger.info(f"完成 (时间: {result.time})")

            except Exception as e:
                error_msg = f"运行 {i+1} 失败: {str(e)}"
                error_details.append(error_msg)

                if self.config.verbose:
                    self.logger.error(f"失败: {e}")

                self.logger.error(error_msg)

        # 解析时间值
        time_values = []
        cpu_time_values = []

        for result in raw_results:
            time_val = self._parse_time_value(result.time)
            cpu_time_val = self._parse_time_value(result.cpu_time)

            if time_val is not None:
                time_values.append(time_val)
            if cpu_time_val is not None:
                cpu_time_values.append(cpu_time_val)

        # 计算统计信息
        time_stats = self._calculate_statistics(time_values)
        cpu_time_stats = self._calculate_statistics(cpu_time_values)

        # 创建结果对象
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            config_used=self.config,
            num_runs=self.config.num_runs,
            successful_runs=successful_runs,
            failed_runs=self.config.num_runs - successful_runs,
            raw_iree_results=raw_results,
            time_values=time_values,
            cpu_time_values=cpu_time_values,
            # 时间统计
            mean_time=time_stats.get("mean", 0),
            median_time=time_stats.get("median", 0),
            min_time=time_stats.get("min", 0),
            max_time=time_stats.get("max", 0),
            std_dev_time=time_stats.get("std_dev"),
            cv_time=time_stats.get("cv"),
            # CPU时间统计
            mean_cpu_time=cpu_time_stats.get("mean", 0),
            median_cpu_time=cpu_time_stats.get("median", 0),
            min_cpu_time=cpu_time_stats.get("min", 0),
            max_cpu_time=cpu_time_stats.get("max", 0),
            std_dev_cpu_time=cpu_time_stats.get("std_dev"),
            cv_cpu_time=cpu_time_stats.get("cv"),
            # 性能指标
            throughput=time_stats.get("throughput", 0),
            cpu_throughput=cpu_time_stats.get("throughput", 0),
            stability_rating=self._get_stability_rating(time_stats.get("cv")),
            error_details=error_details,
        )

        # 清理文件
        if self.config.cleanup_after and saved_files:
            self._cleanup_files(saved_files)

        return result

    def _cleanup_files(self, file_paths: List[str]):
        """Clean up saved files"""
        for file_path in file_paths:
            try:
                os.remove(file_path)
                if self.config.verbose:
                    self.logger.info(f"已清理文件: {file_path}")
            except OSError as e:
                self.logger.warning(f"清理文件失败 {file_path}: {e}")

    def print_results(self, result: BenchmarkResult):
        """Print formatted enhanced benchmark results"""
        print("\n" + "=" * 70)
        print("增强基准测试结果:")
        print("=" * 70)

        print(f"📋 基准测试名称: {result.benchmark_name}")
        print(f"✅ 成功运行: {result.successful_runs}/{result.num_runs}")

        if result.failed_runs > 0:
            print(f"❌ 失败运行: {result.failed_runs}")
            print("🔍 错误详情:")
            for error in result.error_details:
                print(f"   {error}")

        if result.successful_runs == 0:
            print("❌ 所有基准测试运行都失败了")
            return

        print()
        print("⏱️  墙钟时间统计:")
        print(f"   平均值: {result.mean_time:.6f} 秒")
        print(f"   中位数: {result.median_time:.6f} 秒")
        print(f"   最小值: {result.min_time:.6f} 秒")
        print(f"   最大值: {result.max_time:.6f} 秒")

        if result.std_dev_time is not None:
            print(f"   标准差: {result.std_dev_time:.6f} 秒")
            print(f"   变异系数: {result.cv_time*100:.2f}%")

        print()
        print("🖥️  CPU时间统计:")
        print(f"   平均值: {result.mean_cpu_time:.6f} 秒")
        print(f"   中位数: {result.median_cpu_time:.6f} 秒")
        print(f"   最小值: {result.min_cpu_time:.6f} 秒")
        print(f"   最大值: {result.max_cpu_time:.6f} 秒")

        if result.std_dev_cpu_time is not None:
            print(f"   标准差: {result.std_dev_cpu_time:.6f} 秒")
            print(f"   变异系数: {result.cv_cpu_time*100:.2f}%")

        print()
        print("📊 性能指标:")
        print(f"   墙钟吞吐量: {result.throughput:.2f} 次/秒")
        print(f"   CPU吞吐量: {result.cpu_throughput:.2f} 次/秒")
        print(f"   稳定性评级: {result.stability_rating}")

    def print_result_simple(self, result: BenchmarkResult):
        """Print simplified benchmark results focusing on average times"""
        print(
            f"📋 {result.benchmark_name} | ✅ {result.successful_runs}/{result.num_runs}"
        )

        if result.successful_runs == 0:
            print("❌ 所有基准测试运行都失败了")
            return

        print(f"⏱️  平均墙钟时间: {result.mean_time:.6f} 秒")
        print(f"🖥️  平均CPU时间: {result.mean_cpu_time:.6f} 秒")

        if result.failed_runs > 0:
            print(f"❌ 失败运行: {result.failed_runs}")
        print("-" * 50)


class ComparisonBenchmark:
    """performance comparison tools"""

    @staticmethod
    def compare_results(
        results: List[BenchmarkResult], metric: str = "mean_time"
    ) -> Dict[str, Any]:
        """Compare multiplebenchmark results"""
        if not results:
            return {}

        values = []
        names = []

        for result in results:
            if hasattr(result, metric):
                values.append(getattr(result, metric))
                names.append(result.benchmark_name)

        if not values:
            return {}

        # 对于时间指标，越小越好；对于吞吐量指标，越大越好
        is_time_metric = "time" in metric.lower()
        best_idx = values.index(min(values) if is_time_metric else max(values))

        comparison = {
            "metric": metric,
            "results": list(zip(names, values)),
            "best": names[best_idx],
            "best_value": values[best_idx],
            "improvement_ratios": [],
            "is_time_metric": is_time_metric,
        }

        # 计算改进比率
        base_value = values[best_idx]
        for i, value in enumerate(values):
            if value > 0 and base_value > 0:
                if is_time_metric:
                    ratio = value / base_value
                else:
                    ratio = base_value / value
                comparison["improvement_ratios"].append((names[i], ratio))

        return comparison

    @staticmethod
    def print_comparison(comparison: Dict[str, Any]):
        """Print formatted enhanced comparison results"""
        if not comparison:
            print("没有可比较的结果")
            return

        metric_name = comparison["metric"]
        is_time_metric = comparison["is_time_metric"]

        print("\n" + "=" * 70)
        print(f"增强性能比较分析 (基于 {metric_name}):")
        print("=" * 70)

        print(f"🏆 最优结果: {comparison['best']}")
        print(f"   {metric_name}: {comparison['best_value']:.6f}")
        print()

        print("📈 详细比较:")
        sorted_results = sorted(
            comparison["results"], key=lambda x: x[1], reverse=not is_time_metric
        )

        for i, (name, value) in enumerate(sorted_results):
            rank_emoji = (
                "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            )
            print(f"   {rank_emoji} {name}: {value:.6f}")

        if comparison["improvement_ratios"]:
            print(f"\n⚡ 性能比较 (相对于最优 {comparison['best']}):")
            sorted_ratios = sorted(comparison["improvement_ratios"], key=lambda x: x[1])

            for name, ratio in sorted_ratios:
                if ratio == 1.0:
                    print(f"   ✨ {name}: 基准 (1.00x)")
                elif ratio > 1.0:
                    if is_time_metric:
                        print(f"   🐌 {name}: 慢 {ratio:.2f}x")
                    else:
                        print(f"   🚀 {name}: 快 {ratio:.2f}x")
                else:
                    if is_time_metric:
                        print(f"   🚀 {name}: 快 {1/ratio:.2f}x")
                    else:
                        print(f"   🐌 {name}: 慢 {1/ratio:.2f}x")


# 便捷函数
def quick_benchmark(
    module: Union[VmModule, PathLike],
    inputs: List[Union[np.ndarray, str]],
    benchmark_name: str = "quick_test",
    num_runs: int = 10,
    entry_function: Optional[str] = None,
    timeout: Optional[float] = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Quick enhanced benchmark function

    Args:
        module: IREE VmModule or path to module file
        inputs: Input data
        benchmark_name: Name for the benchmark
        num_runs: Number of runs
        entry_function: Entry function name
        timeout: Timeout in seconds
        verbose: Whether to print results

    Returns:
        EnhancedBenchmarkResult
    """
    config = BenchmarkConfig(
        num_runs=num_runs,
        entry_function=entry_function,
        timeout=timeout,
        verbose=verbose,
    )

    runner = BenchmarkRunner(config)
    result = runner.run_benchmark(module, inputs, benchmark_name)

    if verbose:
        runner.print_results(result)

    return result
