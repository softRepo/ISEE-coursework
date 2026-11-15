"""训练与评估可视化工具。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# 使用非交互后端保存图片，避免命令行环境缺少 Tk 时出错。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def setup_chinese_font() -> None:
    """设置 Matplotlib 中文字体兼容选项。"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_training_curves(history: list[dict[str, float]], output_dir: str | Path, model_name: str) -> None:
    """保存 loss 曲线和 accuracy 曲线。"""
    setup_chinese_font()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="训练损失")
    plt.plot(epochs, val_loss, label="验证损失")
    plt.xlabel("轮次")
    plt.ylabel("损失")
    plt.title(f"{model_name} 损失曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"{model_name}_loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="训练准确率")
    plt.plot(epochs, val_acc, label="验证准确率")
    plt.xlabel("轮次")
    plt.ylabel("准确率")
    plt.title(f"{model_name} 准确率曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"{model_name}_accuracy_curve.png", dpi=160)
    plt.close()


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], save_path: str | Path) -> None:
    """保存混淆矩阵图片。"""
    setup_chinese_font()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("混淆矩阵")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
