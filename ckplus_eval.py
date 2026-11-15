"""CK+ 外部泛化评估脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from utils.common import CLASS_NAMES, build_model, ensure_dir, get_device
from utils.dataset import CKPlusTableDataset
from utils.metrics import classification_metrics, confusion_matrix_np
from utils.visualize import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="在 CK+ 表格数据集上测试 FER2013 训练好的模型")
    parser.add_argument("--model", type=str, default="mini_resnet", choices=["simple_cnn", "improved_cnn", "mini_resnet"])
    parser.add_argument("--weights", type=str, default="checkpoints/best_mini_resnet.pth")
    parser.add_argument("--ckplus_path", type=str, default="data/ck_dataset/ckextended.csv", help="CK+ 表格文件路径")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--usage", type=str, default="all", help="可选 all、Training、PublicTest、PrivateTest")
    parser.add_argument("--max_wrong_samples", type=int, default=0, help="最多保存多少个错误分类样例")
    return parser.parse_args()


def resolve_ckplus_path(ckplus_path: str) -> Path:
    """检查并返回 CK+ 表格文件路径。"""
    path = Path(ckplus_path)
    if not path.exists():
        raise FileNotFoundError(f"CK+ 文件不存在：{path}，默认路径应为 data/ck_dataset/ckextended.csv。")
    return path


def normalize_usage(usage: str) -> str | None:
    """将 Usage 参数转换为数据集读取需要的形式。"""
    value = usage.strip()
    if value.lower() == "all":
        return None
    allowed = {"training", "publictest", "privatetest"}
    if value.lower() not in allowed:
        raise ValueError("Usage 参数只能是 all、Training、PublicTest 或 PrivateTest。")
    return value


def load_weights(model: torch.nn.Module, weights_path: str | Path, device: torch.device, model_name: str) -> None:
    """加载 FER2013 训练好的模型权重。"""
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"模型权重不存在：{path}，请先确认 FER2013 训练结果已经保存。")

    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_name" in checkpoint and checkpoint["model_name"] != model_name:
        print(f"提示：权重中记录的模型为 {checkpoint['model_name']}，当前选择的模型为 {model_name}，请确认二者结构一致。")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """将 CK+ tensor 图像转为便于保存的图片。"""
    array = image_tensor.squeeze(0).cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L").resize((144, 144), Image.NEAREST).convert("RGB")


def save_wrong_sample(image_tensor: torch.Tensor, true_label: int, pred_label: int, save_path: Path) -> None:
    """保存带真实标签和预测标签的错误样例。"""
    image = tensor_to_pil(image_tensor)
    draw = ImageDraw.Draw(image)
    text = f"T:{CLASS_NAMES[true_label]} P:{CLASS_NAMES[pred_label]}"
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 18), fill=(255, 255, 255))
    draw.text((3, 3), text, fill=(0, 0, 0), font=font)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def format_ckplus_report(model_name: str, ckplus_path: Path, usage: str, matrix: np.ndarray) -> str:
    """生成 CK+ 外部泛化分类报告。"""
    metrics = classification_metrics(matrix)
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    support = matrix.sum(axis=1)

    lines = []
    lines.append("CK+ 外部泛化测试分类报告")
    lines.append(f"模型名称：{model_name}")
    lines.append(f"Usage 筛选：{usage}")
    lines.append("说明：已删除 emotion=7 的 Contempt 样本，仅保留 0 到 6 类。")
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
    """保存每个类别在 CK+ 上的识别统计。"""
    metrics = classification_metrics(matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("CK+ 各类别识别效果统计\n")
        file.write("说明：统计结果基于删除 Contempt 后的 7 类样本。\n")
        for index, class_name in enumerate(CLASS_NAMES):
            total = int(matrix[index].sum())
            correct = int(matrix[index, index])
            file.write(
                f"{class_name}：样本数 {total}，正确 {correct}，"
                f"Precision {metrics['precision'][index]:.4f}，"
                f"Recall {metrics['recall'][index]:.4f}，"
                f"F1 {metrics['f1'][index]:.4f}\n"
            )


def evaluate_ckplus(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    wrong_dir: Path,
    max_wrong_samples: int,
) -> tuple[list[int], list[int]]:
    """在 CK+ 数据集上预测，并保存错误分类样例。"""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    wrong_count = 0

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            for image, label, pred in zip(images, labels, preds):
                label_int = int(label.item())
                pred_int = int(pred.item())
                all_labels.append(label_int)
                all_preds.append(pred_int)
                if label_int != pred_int and wrong_count < max_wrong_samples:
                    save_wrong_sample(image, label_int, pred_int, wrong_dir / f"wrong_{wrong_count:03d}.png")
                    wrong_count += 1

    return all_labels, all_preds


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    ckplus_path = resolve_ckplus_path(args.ckplus_path)
    usage_filter = normalize_usage(args.usage)
    usage_text = "all" if usage_filter is None else usage_filter

    print("开始加载 CK+ 表格数据集...")
    dataset = CKPlusTableDataset(ckplus_path, usage=usage_filter)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"CK+ 文件：{ckplus_path}")
    print(f"Usage 筛选：{usage_text}")
    print(f"可用样本数：{len(dataset)}，已自动删除 Contempt 类样本。")

    device = get_device()
    model = build_model(args.model).to(device)
    load_weights(model, args.weights, device, args.model)
    print(f"模型权重已加载：{args.weights}")
    print(f"当前设备：{device}")

    prefix = f"ckplus_{args.model}"
    wrong_dir = Path(output_dir) / f"{prefix}_wrong_samples"
    all_labels, all_preds = evaluate_ckplus(model, loader, device, wrong_dir, args.max_wrong_samples)
    matrix = confusion_matrix_np(all_labels, all_preds, len(CLASS_NAMES))
    metrics = classification_metrics(matrix)

    report = format_ckplus_report(args.model, ckplus_path, usage_text, matrix)
    report_path = Path(output_dir) / f"{prefix}_classification_report.txt"
    confusion_path = Path(output_dir) / f"{prefix}_confusion_matrix.png"
    statistics_path = Path(output_dir) / f"{prefix}_class_statistics.txt"

    report_path.write_text(report, encoding="utf-8")
    plot_confusion_matrix(matrix, CLASS_NAMES, confusion_path)
    save_class_statistics(matrix, statistics_path)

    print(report)
    print(f"CK+ Accuracy：{metrics['accuracy']:.4f}")
    print(f"CK+ Macro Precision：{metrics['macro_precision']:.4f}")
    print(f"CK+ Macro Recall：{metrics['macro_recall']:.4f}")
    print(f"CK+ Macro F1-score：{metrics['macro_f1']:.4f}")
    print(f"分类报告已保存：{report_path}")
    print(f"混淆矩阵已保存：{confusion_path}")
    print(f"类别统计已保存：{statistics_path}")
    if args.max_wrong_samples > 0:
        print(f"错误样例已保存到：{wrong_dir}")


if __name__ == "__main__":
    main()
