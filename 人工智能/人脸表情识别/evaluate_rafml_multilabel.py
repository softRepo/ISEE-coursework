"""RAF-ML 多标签混合表情 RGB 模型评估脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import ImageDraw, ImageFont
from torch.utils.data import DataLoader

from models.rgb_mini_resnet import RGBMiniResNet
from utils.common import ensure_dir, get_device
from utils.rafml_dataset import (
    RAFML_LABELS_CN,
    RAFML_LABELS_EN,
    RAFML_NUM_LABELS,
    RAFMLDataset,
    collect_rafml_samples,
    print_rafml_statistics,
    unnormalize_rafml_tensor,
)

DEFAULT_TAG = "rafml_rgb_mini_resnet"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估 RAF-ML 6 维多标签混合表情 RGB 模型")
    parser.add_argument("--data_dir", type=str, default="data/RAF-ML")
    parser.add_argument("--model", type=str, default="rafml_rgb_mini_resnet", choices=["rafml_rgb_mini_resnet", "rgb_mini_resnet"])
    parser.add_argument("--weights", type=str, default="checkpoints/best_rafml_rgb_mini_resnet.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_prediction_samples", type=int, default=0)
    parser.add_argument("--max_wrong_samples", type=int, default=0)
    return parser.parse_args()


def build_rafml_model(model_name: str) -> torch.nn.Module:
    """创建 6 维多标签 RGB MiniResNet。"""
    if model_name in {"rafml_rgb_mini_resnet", "rgb_mini_resnet"}:
        return RGBMiniResNet(num_classes=RAFML_NUM_LABELS)
    raise ValueError("未知模型名称，目前仅支持 rafml_rgb_mini_resnet 或 rgb_mini_resnet。")


def load_weights(model: torch.nn.Module, weights_path: str | Path, device: torch.device, model_name: str) -> None:
    """加载 RAF-ML 多标签模型权重。"""
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"模型权重不存在：{path}，请先运行 train_rafml_multilabel.py 训练模型。")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_name" in checkpoint and checkpoint["model_name"] != model_name:
        print(f"提示：权重中记录的模型为 {checkpoint['model_name']}，当前选择的模型为 {model_name}，请确认二者结构一致。")
    if isinstance(checkpoint, dict) and "num_labels" in checkpoint and checkpoint["num_labels"] != RAFML_NUM_LABELS:
        print("提示：权重中记录的标签维度不是 6，请确认这是 RAF-ML 多标签模型。")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def multilabel_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    """计算 RAF-ML 多标签 micro/macro 指标和每类指标。"""
    preds = (probabilities >= threshold).astype(np.int64)
    targets = labels.astype(np.int64)
    eps = 1e-12

    true_positive = (preds * targets).sum(axis=0).astype(np.float64)
    predicted_positive = preds.sum(axis=0).astype(np.float64)
    actual_positive = targets.sum(axis=0).astype(np.float64)
    precision = true_positive / np.maximum(predicted_positive, eps)
    recall = true_positive / np.maximum(actual_positive, eps)
    f1 = 2 * precision * recall / np.maximum(precision + recall, eps)

    micro_tp = true_positive.sum()
    micro_pred = predicted_positive.sum()
    micro_actual = actual_positive.sum()
    micro_precision = micro_tp / max(micro_pred, eps)
    micro_recall = micro_tp / max(micro_actual, eps)
    micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, eps)

    return {
        "preds": preds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": actual_positive,
        "avg_probability": probabilities.mean(axis=0),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


def format_label_indices(values: np.ndarray) -> str:
    """将多标签 0/1 数组格式化为英文短标签。"""
    indices = [RAFML_LABELS_EN[index] for index, value in enumerate(values.tolist()) if int(value) == 1]
    return ",".join(indices) if indices else "None"


def save_sample_image(
    image_tensor: torch.Tensor,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    probabilities: np.ndarray,
    source_path: str,
    save_path: Path,
) -> None:
    """保存 RAF-ML 预测样例或错误样例。"""
    image = unnormalize_rafml_tensor(image_tensor).resize((190, 190))
    draw = ImageDraw.Draw(image)
    true_text = f"T:{format_label_indices(true_labels)}"
    pred_text = f"P:{format_label_indices(pred_labels)}"
    prob_text = " ".join(f"{label[:2]}:{prob:.2f}" for label, prob in zip(RAFML_LABELS_EN, probabilities))
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 58), fill=(255, 255, 255))
    draw.text((3, 2), true_text[:34], fill=(0, 0, 0), font=font)
    draw.text((3, 18), pred_text[:34], fill=(0, 0, 0), font=font)
    draw.text((3, 34), prob_text[:45], fill=(0, 0, 0), font=font)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def evaluate_and_collect(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    prediction_dir: Path,
    wrong_dir: Path,
    max_prediction_samples: int,
    max_wrong_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """完成 RAF-ML test 预测，并保存部分预测样例和错误样例。"""
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    prediction_count = 0
    wrong_count = 0

    with torch.no_grad():
        for images, labels, image_paths in loader:
            logits = model(images.to(device))
            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= threshold).to(torch.int64)
            for image, label, prob, pred, image_path in zip(images, labels, probs, preds, image_paths):
                label_np = label.numpy().astype(np.int64)
                prob_np = prob.numpy().astype(np.float32)
                pred_np = pred.numpy().astype(np.int64)
                all_labels.append(label_np)
                all_probs.append(prob_np)
                if prediction_count < max_prediction_samples:
                    save_sample_image(
                        image,
                        label_np,
                        pred_np,
                        prob_np,
                        image_path,
                        prediction_dir / f"sample_{prediction_count:03d}.png",
                    )
                    prediction_count += 1
                if not np.array_equal(label_np, pred_np) and wrong_count < max_wrong_samples:
                    save_sample_image(
                        image,
                        label_np,
                        pred_np,
                        prob_np,
                        image_path,
                        wrong_dir / f"wrong_{wrong_count:03d}.png",
                    )
                    wrong_count += 1

    return np.stack(all_labels, axis=0), np.stack(all_probs, axis=0)


def format_report(model_name: str, data_dir: Path, weights_path: Path, threshold: float, metrics: dict[str, object]) -> str:
    """生成 RAF-ML 多标签分类报告。"""
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    support = metrics["support"]
    avg_probability = metrics["avg_probability"]

    lines = []
    lines.append("RAF-ML 多标签混合表情 RGB 模型测试集分类报告")
    lines.append(f"模型名称：{model_name}")
    lines.append(f"sigmoid 阈值：{threshold:.2f}")
    lines.append("说明：RAF-ML 是 6 维 sigmoid 多标签识别，没有 Neutral 类。")
    lines.append(f"Micro Precision：{metrics['micro_precision']:.4f}")
    lines.append(f"Micro Recall：{metrics['micro_recall']:.4f}")
    lines.append(f"Micro F1：{metrics['micro_f1']:.4f}")
    lines.append(f"Macro Precision：{metrics['macro_precision']:.4f}")
    lines.append(f"Macro Recall：{metrics['macro_recall']:.4f}")
    lines.append(f"Macro F1：{metrics['macro_f1']:.4f}")
    lines.append("")
    lines.append(f"{'标签':<10}{'英文名称':<14}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>10}{'AvgProb':>10}")
    for index, class_name in enumerate(RAFML_LABELS_CN):
        lines.append(
            f"{class_name:<10}{RAFML_LABELS_EN[index]:<14}"
            f"{precision[index]:>12.4f}{recall[index]:>12.4f}{f1[index]:>12.4f}"
            f"{int(support[index]):>10}{avg_probability[index]:>10.4f}"
        )
    return "\n".join(lines)


def save_class_statistics(metrics: dict[str, object], output_path: Path) -> None:
    """保存 RAF-ML 每个标签维度的识别统计。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("RAF-ML 多标签混合表情各标签识别效果统计\n")
        file.write("说明：每个标签都是一个独立 sigmoid 二分类维度。\n")
        for index, class_name in enumerate(RAFML_LABELS_CN):
            file.write(
                f"{index} {class_name} / {RAFML_LABELS_EN[index]}："
                f"Support {int(metrics['support'][index])}，"
                f"Precision {metrics['precision'][index]:.4f}，"
                f"Recall {metrics['recall'][index]:.4f}，"
                f"F1 {metrics['f1'][index]:.4f}，"
                f"平均预测概率 {metrics['avg_probability'][index]:.4f}\n"
            )


def main() -> None:
    """评估 RAF-ML 多标签混合表情 RGB 模型。"""
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    tag = args.tag.strip() or DEFAULT_TAG
    threshold = max(0.0, min(float(args.threshold), 1.0))

    print("开始加载 RAF-ML test 数据集...")
    test_samples = collect_rafml_samples(args.data_dir, "test")
    print_rafml_statistics("RAF-ML 测试集", test_samples)
    test_dataset = RAFMLDataset(args.data_dir, "test", samples=test_samples, use_aug=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = get_device()
    model = build_rafml_model(args.model).to(device)
    load_weights(model, args.weights, device, args.model)
    print(f"模型权重已加载：{args.weights}")
    print(f"当前设备：{device}")

    prediction_dir = Path(output_dir) / f"{tag}_prediction_samples"
    wrong_dir = Path(output_dir) / f"{tag}_wrong_samples"
    labels, probabilities = evaluate_and_collect(
        model,
        test_loader,
        device,
        threshold,
        prediction_dir,
        wrong_dir,
        args.max_prediction_samples,
        args.max_wrong_samples,
    )
    metrics = multilabel_metrics(labels, probabilities, threshold)

    report_path = Path(output_dir) / f"{tag}_classification_report.txt"
    statistics_path = Path(output_dir) / f"{tag}_class_statistics.txt"
    report = format_report(args.model, Path(args.data_dir), Path(args.weights), threshold, metrics)
    report_path.write_text(report, encoding="utf-8")
    save_class_statistics(metrics, statistics_path)

    print(report)
    print(f"RAF-ML Micro Precision：{metrics['micro_precision']:.4f}")
    print(f"RAF-ML Micro Recall：{metrics['micro_recall']:.4f}")
    print(f"RAF-ML Micro F1：{metrics['micro_f1']:.4f}")
    print(f"RAF-ML Macro Precision：{metrics['macro_precision']:.4f}")
    print(f"RAF-ML Macro Recall：{metrics['macro_recall']:.4f}")
    print(f"RAF-ML Macro F1：{metrics['macro_f1']:.4f}")
    print(f"分类报告已保存：{report_path}")
    print(f"类别统计已保存：{statistics_path}")
    if args.max_prediction_samples > 0:
        print(f"预测样例已保存到：{prediction_dir}")
    if args.max_wrong_samples > 0:
        print(f"错误样例已保存到：{wrong_dir}")


if __name__ == "__main__":
    main()
