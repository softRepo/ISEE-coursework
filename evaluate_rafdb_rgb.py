"""RAF-DB RGB 模型评估脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import ImageDraw, ImageFont
from torch.utils.data import DataLoader

from models.rgb_mini_resnet import RGBMiniResNet
from utils.common import CLASS_NAMES, ensure_dir, get_device
from utils.metrics import classification_metrics, confusion_matrix_np
from utils.rafdb_dataset import RAFDBRGBDataset, collect_rafdb_samples, print_sample_statistics, unnormalize_rgb_tensor
from utils.visualize import plot_confusion_matrix

EXPERIMENT_NAME = "rafdb_rgb_mini_resnet"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估 RAF-DB RGB 表情识别模型")
    parser.add_argument("--data_dir", type=str, default="data/RAF-DB/DATASET")
    parser.add_argument("--model", type=str, default="rgb_mini_resnet", choices=["rgb_mini_resnet"])
    parser.add_argument("--weights", type=str, default="checkpoints/best_rafdb_rgb_mini_resnet.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_wrong_samples", type=int, default=0)
    parser.add_argument("--max_correct_samples", type=int, default=0)
    parser.add_argument("--tag", type=str, default="", help="可选输出文件名前缀，不填则使用默认 RAF-DB RGB 实验名")
    return parser.parse_args()


def build_rgb_model(model_name: str) -> torch.nn.Module:
    """根据名称创建 RAF-DB RGB 模型。"""
    if model_name == "rgb_mini_resnet":
        return RGBMiniResNet(num_classes=len(CLASS_NAMES))
    raise ValueError("未知模型名称，目前仅支持 rgb_mini_resnet。")


def load_weights(model: torch.nn.Module, weights_path: str | Path, device: torch.device, model_name: str) -> None:
    """加载 RAF-DB RGB 模型权重。"""
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"模型权重不存在：{path}，请先运行 train_rafdb_rgb.py 训练模型。")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_name" in checkpoint and checkpoint["model_name"] != model_name:
        print(f"提示：权重中记录的模型为 {checkpoint['model_name']}，当前选择的模型为 {model_name}，请确认二者结构一致。")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def save_sample_image(image_tensor: torch.Tensor, true_label: int, pred_label: int, source_path: str, save_path: Path) -> None:
    """保存 RAF-DB RGB 正确或错误样例。"""
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
    correct_dir: Path,
    wrong_dir: Path,
    max_correct_samples: int,
    max_wrong_samples: int,
) -> tuple[list[int], list[int]]:
    """完成 RAF-DB RGB test 预测，并保存部分正确和错误样例。"""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    correct_count = 0
    wrong_count = 0

    with torch.no_grad():
        for images, labels, image_paths in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            for image, label, pred, image_path in zip(images, labels, preds, image_paths):
                label_int = int(label.item())
                pred_int = int(pred.item())
                all_labels.append(label_int)
                all_preds.append(pred_int)
                if label_int == pred_int and correct_count < max_correct_samples:
                    save_sample_image(image, label_int, pred_int, image_path, correct_dir / f"correct_{correct_count:03d}.png")
                    correct_count += 1
                elif label_int != pred_int and wrong_count < max_wrong_samples:
                    save_sample_image(image, label_int, pred_int, image_path, wrong_dir / f"wrong_{wrong_count:03d}.png")
                    wrong_count += 1

    return all_labels, all_preds


def format_report(model_name: str, data_dir: Path, matrix: np.ndarray) -> str:
    """生成 RAF-DB RGB 分类报告。"""
    metrics = classification_metrics(matrix)
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    support = matrix.sum(axis=1)

    lines = []
    lines.append("RAF-DB RGB 模型测试集分类报告")
    lines.append(f"模型名称：{model_name}")
    lines.append("说明：模型使用 RAF-DB RGB 彩色图像训练，标签已映射为 FER2013 的统一 7 类顺序。")
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
    """保存 RAF-DB RGB 模型每类识别效果。"""
    metrics = classification_metrics(matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("RAF-DB RGB 模型各类别识别效果统计\n")
        file.write("说明：类别顺序为项目统一的 FER2013 标签顺序。\n")
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

    print("开始加载 RAF-DB RGB test 数据集...")
    test_samples = collect_rafdb_samples(args.data_dir, "test")
    print_sample_statistics("RAF-DB RGB 测试集", test_samples)
    test_dataset = RAFDBRGBDataset(args.data_dir, "test", samples=test_samples, use_aug=False)
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

    output_prefix = args.tag.strip() if args.tag.strip() else EXPERIMENT_NAME
    correct_dir = Path(output_dir) / f"{output_prefix}_correct_samples"
    wrong_dir = Path(output_dir) / f"{output_prefix}_wrong_samples"
    all_labels, all_preds = evaluate_and_collect(
        model,
        test_loader,
        device,
        correct_dir,
        wrong_dir,
        args.max_correct_samples,
        args.max_wrong_samples,
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
    print(f"RAF-DB RGB Accuracy：{metrics['accuracy']:.4f}")
    print(f"RAF-DB RGB Macro Precision：{metrics['macro_precision']:.4f}")
    print(f"RAF-DB RGB Macro Recall：{metrics['macro_recall']:.4f}")
    print(f"RAF-DB RGB Macro F1-score：{metrics['macro_f1']:.4f}")
    print(f"分类报告已保存：{report_path}")
    print(f"混淆矩阵已保存：{confusion_path}")
    print(f"类别统计已保存：{statistics_path}")
    if args.max_correct_samples > 0:
        print(f"正确样例已保存到：{correct_dir}")
    if args.max_wrong_samples > 0:
        print(f"错误样例已保存到：{wrong_dir}")


if __name__ == "__main__":
    main()
