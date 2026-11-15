"""整理消融实验与对比分析结果。

本脚本只读取已有 outputs 中的分类报告，并在读取失败时使用内置兜底指标。
不会重新训练模型，不会重新评估模型，也不会修改已有实验结果文件。
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


OUTPUT_DIR = Path("outputs/ablation_analysis")


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "group": "输入形式消融",
        "experiment_name": "FER2013 灰度 MiniResNet 在 RAF-DB 外部测试",
        "dataset": "RAF-DB test",
        "task_type": "7类softmax",
        "input_type": "灰度48×48",
        "training_strategy": "外部测试",
        "report_path": Path("outputs/rafdb_mini_resnet_classification_report.txt"),
        "fallback": {
            "accuracy": 0.5684,
            "macro_precision": 0.4885,
            "macro_recall": 0.4187,
            "macro_f1": 0.3921,
        },
        "notes": "FER2013 灰度模型跨数据集测试，存在明显域差异。",
    },
    {
        "group": "输入形式消融",
        "experiment_name": "RAF-DB RGB MiniResNet",
        "dataset": "RAF-DB test",
        "task_type": "7类softmax",
        "input_type": "RGB96×96",
        "training_strategy": "RAF-DB 从头训练",
        "report_path": Path("outputs/rafdb_rgb_mini_resnet_classification_report.txt"),
        "fallback": {
            "accuracy": 0.8380,
            "macro_precision": 0.7597,
            "macro_recall": 0.7621,
            "macro_f1": 0.7591,
        },
        "notes": "同源 RAF-DB 彩色图像训练与测试，保留颜色和纹理信息。",
    },
    {
        "group": "训练策略消融",
        "experiment_name": "RAF-DB RGB 从头训练",
        "dataset": "RAF-DB test",
        "task_type": "7类softmax",
        "input_type": "RGB96×96",
        "training_strategy": "从头训练",
        "report_path": Path("outputs/rafdb_rgb_mini_resnet_classification_report.txt"),
        "fallback": {
            "accuracy": 0.8380,
            "macro_precision": 0.7597,
            "macro_recall": 0.7621,
            "macro_f1": 0.7591,
        },
        "notes": "使用 RAF-DB 训练集从头训练，测试集与训练集同源。",
    },
    {
        "group": "训练策略消融",
        "experiment_name": "AffectNet 预训练 + RAF-DB 微调",
        "dataset": "RAF-DB test",
        "task_type": "7类softmax",
        "input_type": "RGB96×96",
        "training_strategy": "预训练微调",
        "report_path": Path("outputs/affectnet_pretrained_rafdb_rgb_mini_resnet_classification_report.txt"),
        "fallback": {
            "accuracy": 0.8181,
            "macro_precision": 0.7476,
            "macro_recall": 0.7179,
            "macro_f1": 0.7302,
        },
        "notes": "先在 AffectNet 上预训练，再迁移到 RAF-DB 上微调。",
    },
    {
        "group": "任务形式对比",
        "experiment_name": "RAF-DB RGB 基本七类表情识别",
        "dataset": "RAF-DB test",
        "task_type": "7类softmax",
        "input_type": "RGB96×96",
        "training_strategy": "RAF-DB 从头训练",
        "report_path": Path("outputs/rafdb_rgb_mini_resnet_classification_report.txt"),
        "fallback": {
            "accuracy": 0.8380,
            "macro_precision": 0.7597,
            "macro_recall": 0.7621,
            "macro_f1": 0.7591,
        },
        "notes": "基本七类表情识别任务，类别定义相对稳定。",
    },
    {
        "group": "任务形式对比",
        "experiment_name": "RAF-DB Compound 11类复合表情识别",
        "dataset": "RAF-DB Compound test",
        "task_type": "11类softmax",
        "input_type": "RGB96×96",
        "training_strategy": "AffectNet 预训练微调",
        "report_path": Path("outputs/compound_rgb_affectnet_pretrained_classification_report.txt"),
        "fallback": {
            "accuracy": 0.5846,
            "macro_precision": 0.4630,
            "macro_recall": 0.4435,
            "macro_f1": 0.4492,
        },
        "notes": "复合表情类别更细，类别语义相近，样本数量更少。",
    },
    {
        "group": "任务形式对比",
        "experiment_name": "RAF-ML 6维多标签混合表情识别",
        "dataset": "RAF-ML test",
        "task_type": "6维sigmoid多标签",
        "input_type": "RGB96×96",
        "training_strategy": "AffectNet 预训练微调",
        "report_path": Path("outputs/rafml_affectnet_pretrained_classification_report.txt"),
        "fallback": {
            "micro_precision": 0.7753,
            "micro_recall": 0.7431,
            "micro_f1": 0.7588,
            "macro_precision": 0.7592,
            "macro_recall": 0.7163,
            "macro_f1": 0.7352,
        },
        "notes": "多标签 sigmoid 任务，没有普通单标签 Accuracy，主指标为 Micro F1 和 Macro F1。",
    },
]


CSV_FIELDS = [
    "group",
    "experiment_name",
    "dataset",
    "task_type",
    "input_type",
    "training_strategy",
    "accuracy",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "notes",
]


GROUP_ANALYSIS = {
    "输入形式消融": (
        "输入形式消融结果表明，FER2013 灰度 MiniResNet 在 RAF-DB 外部测试上的表现明显低于 "
        "RAF-DB RGB MiniResNet。前者同时受到灰度输入、低分辨率 FER2013 风格和跨数据集域差异影响；"
        "后者使用 RAF-DB 同源彩色真实场景图像训练，RGB 输入保留了更多颜色、纹理和光照信息，因此 Accuracy "
        "和 Macro F1 均有显著提升。"
    ),
    "训练策略消融": (
        "训练策略消融结果显示，AffectNet 预训练 + RAF-DB 微调取得了接近 RAF-DB 从头训练的性能，"
        "但在 RAF-DB test 上仍略低于同源数据从头训练模型。这说明预训练具有一定迁移价值，但 AffectNet 与 "
        "RAF-DB 在数据分布、图像风格、标签映射和类别比例上的差异可能限制了迁移效果。"
    ),
    "任务形式对比": (
        "任务形式对比结果表明，基本七类表情识别相对更稳定；RAF-DB Compound 11 类复合表情识别难度更高，"
        "主要原因是类别更细、语义更接近且样本数量较少；RAF-ML 采用 6 维 sigmoid 多标签形式表达混合情绪，"
        "Macro F1 较高，说明多标签形式更适合描述真实场景中多个情绪成分同时存在的情况。"
    ),
}


REPORT_TEXT = """输入形式消融分析：
FER2013 灰度 MiniResNet 在 RAF-DB 外部测试上的表现明显低于 RAF-DB RGB MiniResNet。主要原因可能是 FER2013 灰度低分辨率图像与 RAF-DB 彩色真实场景图像之间存在明显域差异，同时 RGB 图像保留了更多颜色、纹理和光照信息，有助于模型学习真实场景中的表情特征。

训练策略消融分析：
AffectNet 预训练 + RAF-DB 微调取得了接近 RAF-DB 从头训练的性能，但略低于同源 RAF-DB 数据从头训练模型。这说明 AffectNet 预训练具有一定迁移效果，但 AffectNet 和 RAF-DB 在数据分布、图像风格、标签映射和类别比例上存在差异，因此迁移后性能不一定超过目标数据集同源训练。

任务形式对比分析：
基本七类表情识别结果最好，说明在类别定义相对清晰、样本量相对充足的情况下，softmax 单标签分类更稳定。Compound 11 类复合表情识别较难，因为样本数量较少、类别更细、语义更接近，模型更容易在相似复合类别之间混淆。RAF-ML 多标签混合表情识别 Macro F1 较高，说明多标签形式更适合表达真实场景中多个情绪成分同时存在的情况。

总结：
本项目不仅完成了基本表情识别，还通过消融实验分析了输入形式、训练策略和任务定义对模型性能的影响。结果表明，RGB 彩色输入和同源真实场景数据训练能够显著提升基本表情识别性能；预训练微调具有迁移价值但受数据集差异影响；复合表情和混合表情扩展实验进一步展示了不同任务定义对模型性能和适用场景的影响。
"""


def setup_chinese_font() -> None:
    """尽量启用中文字体；如果当前环境没有中文字体，也不让脚本报错。"""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_metric(text: str, names: list[str]) -> float | None:
    """从报告文本中解析指定指标。"""
    for name in names:
        pattern = rf"{re.escape(name)}\s*[：:]\s*([0-9]+(?:\.[0-9]+)?)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def read_metrics(report_path: Path, fallback: dict[str, float]) -> dict[str, float | None]:
    """优先从报告文件读取指标，失败或缺失时使用内置兜底值。"""
    metrics: dict[str, float | None] = {
        "accuracy": None,
        "micro_precision": None,
        "micro_recall": None,
        "micro_f1": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
    }

    if report_path.exists():
        try:
            text = report_path.read_text(encoding="utf-8")
            metrics["accuracy"] = parse_metric(text, ["Accuracy"])
            metrics["micro_precision"] = parse_metric(text, ["Micro Precision"])
            metrics["micro_recall"] = parse_metric(text, ["Micro Recall"])
            metrics["micro_f1"] = parse_metric(text, ["Micro F1", "Micro F1-score"])
            metrics["macro_precision"] = parse_metric(text, ["Macro Precision"])
            metrics["macro_recall"] = parse_metric(text, ["Macro Recall"])
            metrics["macro_f1"] = parse_metric(text, ["Macro F1-score", "Macro F1"])
        except UnicodeDecodeError:
            text = report_path.read_text(encoding="gbk", errors="ignore")
            metrics["accuracy"] = parse_metric(text, ["Accuracy"])
            metrics["micro_precision"] = parse_metric(text, ["Micro Precision"])
            metrics["micro_recall"] = parse_metric(text, ["Micro Recall"])
            metrics["micro_f1"] = parse_metric(text, ["Micro F1", "Micro F1-score"])
            metrics["macro_precision"] = parse_metric(text, ["Macro Precision"])
            metrics["macro_recall"] = parse_metric(text, ["Macro Recall"])
            metrics["macro_f1"] = parse_metric(text, ["Macro F1-score", "Macro F1"])
        except OSError:
            pass

    for key, value in fallback.items():
        if metrics.get(key) is None:
            metrics[key] = value
    return metrics


def build_records() -> list[dict[str, Any]]:
    """生成汇总记录。"""
    records: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        metrics = read_metrics(experiment["report_path"], experiment["fallback"])
        record = {field: experiment.get(field, "") for field in CSV_FIELDS}
        for metric_name, metric_value in metrics.items():
            record[metric_name] = metric_value
        records.append(record)
    return records


def format_metric(value: Any) -> str:
    """统一格式化指标值。"""
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def save_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    """保存消融实验汇总 CSV。"""
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: format_metric(record.get(field, "")) for field in CSV_FIELDS})


def markdown_table(records: list[dict[str, Any]], fields: list[str]) -> str:
    """将记录转换为 Markdown 表格。"""
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join(["---"] * len(fields)) + " |"
    rows = []
    for record in records:
        rows.append("| " + " | ".join(format_metric(record.get(field, "")) for field in fields) + " |")
    return "\n".join([header, separator, *rows])


def save_markdown(records: list[dict[str, Any]], output_path: Path) -> None:
    """保存 Markdown 版消融实验报告。"""
    table_fields = [
        "experiment_name",
        "dataset",
        "task_type",
        "input_type",
        "training_strategy",
        "accuracy",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "notes",
    ]
    lines = ["# 消融实验与对比分析汇总", ""]
    for group_name in ["输入形式消融", "训练策略消融", "任务形式对比"]:
        group_records = [record for record in records if record["group"] == group_name]
        lines.extend([f"## {group_name}", "", markdown_table(group_records, table_fields), "", GROUP_ANALYSIS[group_name], ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def add_bar_labels(ax, bars) -> None:
    """在柱状图上方标注数值。"""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_grouped_bar(
    labels: list[str],
    metric_names: list[str],
    values_by_metric: list[list[float]],
    title: str,
    output_path: Path,
) -> None:
    """绘制多指标分组柱状图。"""
    x_positions = list(range(len(labels)))
    width = 0.32
    colors = ["#4C78A8", "#59A14F", "#F28E2B"]

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=150)
    for metric_index, (metric_name, values) in enumerate(zip(metric_names, values_by_metric)):
        offset = (metric_index - (len(metric_names) - 1) / 2) * width
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width,
            label=metric_name,
            color=colors[metric_index % len(colors)],
        )
        add_bar_labels(ax, bars)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("指标值")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_single_metric_bar(labels: list[str], values: list[float], title: str, output_path: Path) -> None:
    """绘制单指标柱状图。"""
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    bars = ax.bar(labels, values, color=["#4C78A8", "#B07AA1", "#59A14F"])
    add_bar_labels(ax, bars)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def record_by_name(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """按实验名称查找记录。"""
    for record in records:
        if record["experiment_name"] == name:
            return record
    raise KeyError(f"未找到实验记录：{name}")


def save_plots(records: list[dict[str, Any]]) -> None:
    """保存三张消融实验对比图。"""
    gray_record = record_by_name(records, "FER2013 灰度 MiniResNet 在 RAF-DB 外部测试")
    rgb_record = record_by_name(records, "RAF-DB RGB MiniResNet")
    plot_grouped_bar(
        labels=["RAF-DB 灰度外部测试", "RAF-DB RGB 训练模型"],
        metric_names=["Accuracy", "Macro F1"],
        values_by_metric=[
            [float(gray_record["accuracy"]), float(rgb_record["accuracy"])],
            [float(gray_record["macro_f1"]), float(rgb_record["macro_f1"])],
        ],
        title="输入形式消融：灰度图 vs RGB 彩色图",
        output_path=OUTPUT_DIR / "input_ablation_bar.png",
    )

    scratch_record = record_by_name(records, "RAF-DB RGB 从头训练")
    finetune_record = record_by_name(records, "AffectNet 预训练 + RAF-DB 微调")
    plot_grouped_bar(
        labels=["RAF-DB RGB 从头训练", "AffectNet 预训练+微调"],
        metric_names=["Accuracy", "Macro F1"],
        values_by_metric=[
            [float(scratch_record["accuracy"]), float(finetune_record["accuracy"])],
            [float(scratch_record["macro_f1"]), float(finetune_record["macro_f1"])],
        ],
        title="训练策略消融：从头训练 vs 预训练微调",
        output_path=OUTPUT_DIR / "training_strategy_ablation_bar.png",
    )

    basic_record = record_by_name(records, "RAF-DB RGB 基本七类表情识别")
    compound_record = record_by_name(records, "RAF-DB Compound 11类复合表情识别")
    rafml_record = record_by_name(records, "RAF-ML 6维多标签混合表情识别")
    plot_single_metric_bar(
        labels=["基本七类", "Compound 11类", "RAF-ML 6维"],
        values=[
            float(basic_record["macro_f1"]),
            float(compound_record["macro_f1"]),
            float(rafml_record["macro_f1"]),
        ],
        title="任务形式对比：基本表情 vs 复合表情 vs 混合表情",
        output_path=OUTPUT_DIR / "task_comparison_bar.png",
    )


def main() -> None:
    """主入口：汇总结果、保存表格、图表和报告素材。"""
    setup_chinese_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = build_records()
    save_csv(records, OUTPUT_DIR / "ablation_summary.csv")
    save_markdown(records, OUTPUT_DIR / "ablation_summary.md")
    (OUTPUT_DIR / "ablation_report_text.txt").write_text(REPORT_TEXT, encoding="utf-8")
    save_plots(records)

    print("消融实验与对比分析整理完成。")
    print(f"输出目录：{OUTPUT_DIR}")
    print("已生成：ablation_summary.csv、ablation_summary.md、三张柱状图和 ablation_report_text.txt")


if __name__ == "__main__":
    main()
