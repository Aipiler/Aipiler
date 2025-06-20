import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_csv(type: str, seq_len: list[int], einlang: list[int], torch_mlir: list[int]):
    # 创建一个字典来组织数据
    data = {"seq_len": seq_len, "einlang": einlang, "torch_mlir": torch_mlir}

    # 使用pandas创建一个DataFrame
    df = pd.DataFrame(data)

    # 将DataFrame导出为CSV文件，index=False表示不将索引写入文件
    df.to_csv(f"csv/{type}_data.csv", index=False)

    print(f"CSV file '{type}_data.csv' has been created successfully.")
    print(df.to_csv(index=False))


def cal_data(type: str, seq_len: list[int], einlang: list[int], torch_mlir: list[int]):

    # 计算性能提升倍数
    speedup_ratios = [torch_mlir[i] / einlang[i] for i in range(len(seq_len))]
    print(f"\n{type} 性能提升倍数 (torch-mlir / einlang):")
    for i, ratio in enumerate(speedup_ratios):
        print(f"seq_len {seq_len[i]}: {ratio:.2f}x faster")

    # 计算平均性能提升
    avg_speedup = sum(speedup_ratios) / len(speedup_ratios)
    print(f"\n{type} 平均性能提升: {avg_speedup:.2f}x")


def draw(type: str, seq_len: list[int], einlang: list[int], torch_mlir: list[int]):

    # 设置matplotlib参数以符合学术论文标准
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["grid.alpha"] = 0.3

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))

    # 设置柱状图的位置
    x = np.arange(len(seq_len))
    width = 0.30  # 柱子宽度

    # 绘制柱状图
    bars1 = ax.bar(
        x - width / 2,
        einlang,
        width,
        label="Einlang",
        color="#ff7f0e",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        torch_mlir,
        width,
        label="torch-mlir",
        color="#3a70ed",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )

    # 设置坐标轴
    ax.set_xlabel("Sequence Length", fontsize=14, fontweight="bold")
    # 在y轴标签中明确标注使用了对数刻度
    ax.set_ylabel("Execution Time (ms) - Log Scale", fontsize=14, fontweight="bold")

    # 设置x轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seq_len])

    # 使用对数刻度y轴以更好地显示数据差异
    ax.set_yscale("log")

    # 添加网格（只显示y轴网格）
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # 设置图例
    ax.legend(
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=12,
        framealpha=0.9,
    )

    # 设置标题
    # ax.set_title(
    #     "Performance Comparison: RMSNorm + Matmul (Einlang vs torch-mlir)",
    #     fontsize=16,
    #     fontweight="bold",
    #     pad=20,
    # )

    # 在柱子上添加数值标签（可选）
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
                rotation=0,
            )

    # 添加数值标签
    add_value_labels(bars1)
    add_value_labels(bars2)

    # 调整布局
    plt.tight_layout()

    # 设置图表边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # 显示图表
    plt.show()

    # 如果需要保存为高质量图片（推荐用于论文）
    plt.savefig(
        f"pic/performance_comparison_bar_{type}.pdf", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        f"pic/performance_comparison_bar_{type}.png", dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    # 数据

    data = {}

    # cpu-static: 3-5x提升
    seq_len = [16, 32, 64, 128, 256, 512, 1024]
    einlang = [1.832, 3.610, 7.774, 14.420, 27.380, 52.920, 108.725]
    torch_mlir = [9.470, 14.120, 22.925, 41.630, 81.185, 154.450, 308.400]

    data["x86_static"] = (seq_len, einlang, torch_mlir)
    # cpu-dyn
    # 新的性能测试数据
    seq_len = [16, 32, 64, 128, 256, 512, 1024]
    einlang = [2.004, 3.914, 8.181, 15.125, 30.720, 55.230, 114.745]
    torch_mlir = [10.035, 15.095, 25.500, 42.190, 79.410, 155.600, 307.700]
    data["x86_dyn"] = (seq_len, einlang, torch_mlir)

    # cuda-static
    seq_len = [16, 32, 64, 128, 256, 512]  # , 1024
    einlang = [
        0.111,
        0.112,
        0.125,
        0.148,
        0.208,
        0.333,
    ]  # 1024处没有数据，用None表示
    torch_mlir = [2.491, 0.418, 0.426, 0.699, 1.453, 2.310]  # , 4.025
    data["cuda_static"] = (seq_len, einlang, torch_mlir)

    # cuda-static
    seq_len = [16, 32, 64, 128, 256, 512, 1024]
    einlang = [0.959, 1.734, 3.207, 6.110, 11.900, 23.500, 46.655]
    torch_mlir = [1.249, 2.444, 4.738, 9.121, 18.100, 36.150, 71.780]
    data["cuda_dyn"] = (seq_len, einlang, torch_mlir)

    for t, d in data.items():
        cal_data(t, d[0], d[1], d[2])
        draw(t, d[0], d[1], d[2])
        save_csv(t, d[0], d[1], d[2])
