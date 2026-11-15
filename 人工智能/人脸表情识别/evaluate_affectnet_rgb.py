"""AffectNet RGB 预训练模型正式评估脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import ImageDraw, ImageFont
from torch.utils.data import DataLoader

from models.rgb_mini_resnet import RGBMiniResNet
from utils.affectnet_dataset import AffectNetYOLODataset, collect_affectnet_samples, print_affectnet_statistics
from utils.common import CLASS_NAMES, ensure_dir, get_device
from utils.metrics import classification_metrics, confusion_matrix_np
from utils.rafdb_dataset import unnormalize_rgb_tensor
from utils.visualize import plot_confusion_matrix

EXPERIMENT_NAME = "affectnet_rgb_mini_resnet"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="在 AffectNet test set 上评估 RGBMiniResNet 预训练模型")
    parser.add_argument("--data_dir", type=str, default="data/AffectNet/YOLO_format")
    parser.add_argument("--model", type=str, default="rgb_mini_resnet", choices=["rgb_mini_resnet"])
    parser.add_argument("--weights", type=str, default="checkpoints/best_affectnet_rgb_mini_resnet.pth")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_wrong_samples", type=int, default=0)
    parser.add_argument("--max_correct_samples", type=int, default=0)
    parser.add_argument("--tag", type=str, default="", help="可选输出文件名前缀，不填则使用默认 AffectNet 实验名")
    return parser.parse_args()


def build_rgb_model(model_name: str) -> torch.nn.Module:
    """根据模型名称创建 RGBMiniResNet。"""
    if model_name == "rgb_mini_resnet":
        return RGBMiniResNet(num_classes=len(CLASS_NAMES))
    raise ValueError("未知模型名称，目前仅支持 rgb_mini_resnet。")


def load_weights(model: torch.nn.Module, weights_path: str | Path, device: torch.device, model_name: str) -> None:
    """加载 AffectNet RGB 预训练模型权重。"""
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"模型权重不存在：{path}，请先确认 AffectNet 预训练权重已经保存。")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_name" in checkpoint and checkpoint["model_name"] != model_name:
        print(f"提示：权重中记录的模型为 {checkpoint['model_name']}，当前选择的模型为 {model_name}，请确认二者结构一致。")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def save_wrong_sample(image_tensor: torch.Tensor, true_label: int, pred_label: int, source_path: str, save_path: Path) -> None:
    """保存 AffectNet 错误分类样例。"""
    image = unnormalize_rgb_tensor(image_tensor).resize((160, 160))
    draw = ImageDraw.Draw(image)
    text = f"T:{CLASS_NAMES[true_label]} P:{CLASS_NAMES[pred_label]}"
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 34), fill=(255, 255, 255))
    draw.text((3, 3), text, fill=(0, 0, 0), font=font)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def evaluate_and_collect(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    wrong_dir: Path,
    max_wrong_samples: int,
    correct_dir: Path | None = None,
    max_correct_samples: int = 0,
) -> tuple[list[int], list[int]]:
    """完成 AffectNet test 预测，并保存部分错误样例。"""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    wrong_count = 0
    correct_count = 0

    with torch.no_grad():
        for images, labels, image_paths in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            for image, label, pred, image_path in zip(images, labels, preds, image_paths):
                label_int = int(label.item())
                pred_int = int(pred.item())
                all_labels.append(label_int)
                all_preds.append(pred_int)
                if label_int == pred_int and correct_dir is not None and correct_count < max_correct_samples:
                    save_wrong_sample(image, label_int, pred_int, image_path, correct_dir / f"correct_{correct_count:03d}.png")
                    correct_count += 1
                elif label_int != pred_int and wrong_count < max_wrong_samples:
                    save_wrong_sample(image, label_int, pred_int, image_path, wrong_dir / f"wrong_{wrong_count:03d}.png")
                    wrong_count += 1

    return all_labels, all_preds


def format_report(model_name: str, data_dir: Path, matrix: np.ndarray) -> str:
    """生成 AffectNet test set 分类报告。"""
    metrics = classification_metrics(matrix)
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    support = matrix.sum(axis=1)

    lines = []
    lines.append("AffectNet RGB 预训练模型 test set 分类报告")
    lines.append(f"模型名称：{model_name}")
    lines.append("说明：已删除 class_id=1 的 Contempt 类，剩余类别已映射到项目统一 FER2013 七类顺序。")
    lines.append(f"Accuracy：{metrics['accuracy']:.4f}")
    lines.append(f"Macro Precision：{metrics['macro_precision']:.4f}")
    lines.append(f"Macro Recall：{metrics['macro_recall']:.4f}")
    lines.append(f"Macro F1-score：{metrics['macro_f1']:.4f}")
    lines.append("")
    lines.append(f"{'类别':<12}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>10}")
    for index, class_name in enumerate(CLASS_NAMES):
        lines.append(
            f"{class_name:<12}{precision[index]:>12.4f}{recall[index]:>12.4f}{f1[index]:>12.4f}{int(support[index]):>10}"
        )
    return "\n".join(lines)


def save_class_statistics(matrix: np.ndarray, output_path: Path) -> None:
    """保存 AffectNet test set 每类识别统计。"""
    metrics = classification_metrics(matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("AffectNet RGB 预训练模型各类别识别效果统计\n")
        file.write("说明：类别顺序为项目统一的 FER2013 标签顺序，Contempt 类已删除。\n")
        for index, class_name in enumerate(CLASS_NAMES):
            total = int(matrix[index].sum())
            correct = int(matrix[index, index])
            file.write(
                f"{class_name}：样本数 {total}，正确 {correct}，"
                f"Precision {metrics['precision'][index]:.4f}，"
                f"Recall {metrics['recall'][index]:.4f}，"
                f"F1 {metrics['f1'][index]:.4f}\n"
            )


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    output_prefix = args.tag.strip() if args.tag.strip() else EXPERIMENT_NAME

    print("开始加载 AffectNet test set...")
    test_samples = collect_affectnet_samples(args.data_dir, "test", max_per_class=0)
    print_affectnet_statistics("AffectNet test set", test_samples)
    test_dataset = AffectNetYOLODataset(test_samples, use_aug=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = get_device()
    model = build_rgb_model(args.model).to(device)
    load_weights(model, args.weights, device, args.model)
    print(f"模型权重已加载：{args.weights}")
    print(f"当前设备：{device}")

    wrong_dir = Path(output_dir) / f"{output_prefix}_wrong_samples"
    correct_dir = Path(output_dir) / f"{output_prefix}_correct_samples" if args.tag.strip() else None
    all_labels, all_preds = evaluate_and_collect(
        model,
        test_loader,
        device,
        wrong_dir,
        args.max_wrong_samples,
        correct_dir,
        args.max_correct_samples,
    )
    matrix = confusion_matrix_np(all_labels, all_preds, len(CLASS_NAMES))
    metrics = classification_metrics(matrix)

    report = format_report(args.model, Path(args.data_dir), matrix)
    report_path = Path(output_dir) / f"{output_prefix}_classification_report.txt"
    confusion_path = Path(output_dir) / f"{output_prefix}_confusion_matrix.png"
    statistics_path = Path(output_dir) / f"{output_prefix}_class_statistics.txt"

    report_path.write_text(report, encoding="utf-8")
    plot_confusion_matrix(matrix, CLASS_NAMES, confusion_path)
    save_class_statistics(matrix, statistics_path)

    print(report)
    print(f"AffectNet Accuracy：{metrics['accuracy']:.4f}")
    print(f"AffectNet Macro Precision：{metrics['macro_precision']:.4f}")
    print(f"AffectNet Macro Recall：{metrics['macro_recall']:.4f}")
    print(f"AffectNet Macro F1-score：{metrics['macro_f1']:.4f}")
    print(f"分类报告已保存：{report_path}")
    print(f"混淆矩阵已保存：{confusion_path}")
    print(f"类别统计已保存：{statistics_path}")
    if args.max_wrong_samples > 0:
        print(f"错误样例已保存到：{wrong_dir}")
    if correct_dir is not None:
        if args.max_correct_samples > 0:
            print(f"正确样例已保存到：{correct_dir}")


if __name__ == "__main__":
    main()
