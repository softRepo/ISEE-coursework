"""分类指标计算工具。"""

from __future__ import annotations

import numpy as np


def confusion_matrix_np(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray, num_classes: int) -> np.ndarray:
    """计算混淆矩阵。"""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def classification_metrics(matrix: np.ndarray) -> dict[str, np.ndarray | float]:
    """根据混淆矩阵计算 accuracy、precision、recall、f1。"""
    eps = 1e-12
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_total = matrix.sum(axis=0).astype(np.float64)
    actual_total = matrix.sum(axis=1).astype(np.float64)

    precision = true_positive / np.maximum(predicted_total, eps)
    recall = true_positive / np.maximum(actual_total, eps)
    f1 = 2 * precision * recall / np.maximum(precision + recall, eps)
    accuracy = true_positive.sum() / max(matrix.sum(), 1)
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    return {
        "accuracy": float(accuracy),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def format_classification_report(class_names: list[str], matrix: np.ndarray) -> str:
    """生成中文分类报告文本。"""
    metrics = classification_metrics(matrix)
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    support = matrix.sum(axis=1)

    lines = []
    lines.append("FER2013 测试集分类报告")
    lines.append(f"总体准确率：{metrics['accuracy']:.4f}")
    lines.append("")
    lines.append(f"{'类别':<12}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>10}")
    for index, class_name in enumerate(class_names):
        lines.append(
            f"{class_name:<12}{precision[index]:>12.4f}{recall[index]:>12.4f}{f1[index]:>12.4f}{int(support[index]):>10}"
        )
    lines.append("")
    lines.append(f"宏平均 Precision：{metrics['macro_precision']:.4f}")
    lines.append(f"宏平均 Recall：{metrics['macro_recall']:.4f}")
    lines.append(f"宏平均 F1-score：{metrics['macro_f1']:.4f}")
    return "\n".join(lines)
