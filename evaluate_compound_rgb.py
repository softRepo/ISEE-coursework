"""RAF-DB Compound 复合表情 RGB 模型评估脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import ImageDraw, ImageFont
from torch.utils.data import DataLoader

from models.rgb_mini_resnet import RGBMiniResNet
from utils.common import ensure_dir, get_device
from utils.compound_dataset import (
    COMPOUND_LABELS_CN,
    COMPOUND_LABELS_EN,
    COMPOUND_NUM_CLASSES,
    RAFCompoundDataset,
    collect_compound_samples,
    print_compound_statistics,
    unnormalize_compound_tensor,
)
from utils.metrics import classification_metrics, confusion_matrix_np
from utils.visualize import plot_confusion_matrix

DEFAULT_TAG = "compound_rgb_mini_resnet"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估 RAF-DB Compound 11 类复合表情 RGB 模型")
    parser.add_argument("--data_dir", type=str, default="data/compound")
    parser.add_argument(
        "--model",
        type=str,
        default="compound_rgb_mini_resnet",
        choices=["compound_rgb_mini_resnet", "rgb_mini_resnet"],
    )
    parser.add_argument("--weights", type=str, default="checkpoints/best_compound_rgb_mini_resnet.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_wrong_samples", type=int, default=0)
    parser.add_argument("--max_correct_samples", type=int, default=0)
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG, help="输出文件名前缀")
    return parser.parse_args()


def build_compound_model(model_name: str) -> torch.nn.Module:
    """创建 11 类复合表情 RGB MiniResNet。"""
    if model_name in {"compound_rgb_mini_resnet", "rgb_mini_resnet"}:
        return RGBMiniResNet(num_classes=COMPOUND_NUM_CLASSES)
    raise ValueError("未知模型名称，目前仅支持 compound_rgb_mini_resnet 或 rgb_mini_resnet。")


def load_weights(model: torch.nn.Module, weights_path: str | Path, device: torch.device, model_name: str) -> None:
    """加载复合表情模型权重。"""
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"模型权重不存在：{path}，请先运行 train_compound_rgb.py 训练模型。")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_name" in checkpoint and checkpoint["model_name"] != model_name:
        print(f"提示：权重中记录的模型为 {checkpoint['model_name']}，当前选择的模型为 {model_name}，请确认二者结构一致。")
    if isinstance(checkpoint, dict) and "num_classes" in checkpoint and checkpoint["num_classes"] != COMPOUND_NUM_CLASSES:
        print("提示：权重中记录的类别数不是 11，请确认这是 RAF-DB Compound 复合表情模型。")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def save_sample_image(image_tensor: torch.Tensor, true_label: int, pred_label: int, source_path: str, save_path: Path) -> None:
    """保存复合表情正确或错误样例。"""
    image = unnormalize_compound_tensor(image_tensor).resize((180, 180))
    draw = ImageDraw.Draw(image)
    # 样例图上使用英文短标签，避免部分环境缺少中文 PIL 字体导致保存失败。
    text = f"T:{true_label}-{COMPOUND_LABELS_EN[true_label][:16]} P:{pred_label}-{COMPOUND_LABELS_EN[pred_label][:16]}"
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 42), fill=(255, 255, 255))
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
    """完成 RAF-DB Compound test 预测，并保存部分正确和错误样例。"""
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


def format_report(model_name: str, data_dir: Path, weights_path: Path, matrix: np.ndarray) -> str:
    """生成 RAF-DB Compound 分类报告。"""
    metrics = classification_metrics(matrix)
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    support = matrix.sum(axis=1)

    lines = []
    lines.append("RAF-DB Compound 复合表情 RGB 模型测试集分类报告")
    lines.append(f"模型名称：{model_name}")
    lines.append("说明：本实验为 11 类复合表情识别扩展实验，不替代原有 7 类基本表情识别。")
    lines.append(f"Accuracy：{metrics['accuracy']:.4f}")
    lines.append(f"Macro Precision：{metrics['macro_precision']:.4f}")
    lines.append(f"Macro Recall：{metrics['macro_recall']:.4f}")
    lines.append(f"Macro F1-score：{metrics['macro_f1']:.4f}")
    lines.append("")
    lines.append(f"{'类别':<18}{'英文名称':<28}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>10}")
    for index, class_name_cn in enumerate(COMPOUND_LABELS_CN):
        lines.append(
            f"{class_name_cn:<18}{COMPOUND_LABELS_EN[index]:<28}"
            f"{precision[index]:>12.4f}{recall[index]:>12.4f}{f1[index]:>12.4f}{int(support[index]):>10}"
        )
    return "\n".join(lines)


def save_class_statistics(matrix: np.ndarray, output_path: Path) -> None:
    """保存复合表情每类识别效果。"""
    metrics = classification_metrics(matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("RAF-DB Compound 复合表情各类别识别效果统计\n")
        file.write("说明：标签已从原始 1-11 映射为训练使用的 0-10。\n")
        for index, class_name in enumerate(COMPOUND_LABELS_CN):
            total = int(matrix[index].sum())
            correct = int(matrix[index, index])
            file.write(
                f"{index} {class_name} / {COMPOUND_LABELS_EN[index]}："
                f"样本数 {total}，正确 {correct}，"
                f"Precision {metrics['precision'][index]:.4f}，"
                f"Recall {metrics['recall'][index]:.4f}，"
                f"F1 {metrics['f1'][index]:.4f}\n"
            )


def main() -> None:
    """评估 RAF-DB Compound 复合表情 RGB 模型。"""
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    tag = args.tag.strip() or DEFAULT_TAG

    print("开始加载 RAF-DB Compound test 数据集...")
    test_samples = collect_compound_samples(args.data_dir, "test")
    print_compound_statistics("RAF-DB Compound 测试集", test_samples)
    test_dataset = RAFCompoundDataset(args.data_dir, "test", samples=test_samples, use_aug=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = get_device()
    model = build_compound_model(args.model).to(device)
    load_weights(model, args.weights, device, args.model)
    print(f"模型权重已加载：{args.weights}")
    print(f"当前设备：{device}")

    correct_dir = Path(output_dir) / f"{tag}_correct_samples"
    wrong_dir = Path(output_dir) / f"{tag}_wrong_samples"
    all_labels, all_preds = evaluate_and_collect(
        model,
        test_loader,
        device,
        correct_dir,
        wrong_dir,
        args.max_correct_samples,
        args.max_wrong_samples,
    )
    matrix = confusion_matrix_np(all_labels, all_preds, COMPOUND_NUM_CLASSES)
    metrics = classification_metrics(matrix)

    report_path = Path(output_dir) / f"{tag}_classification_report.txt"
    confusion_path = Path(output_dir) / f"{tag}_confusion_matrix.png"
    statistics_path = Path(output_dir) / f"{tag}_class_statistics.txt"

    report = format_report(args.model, Path(args.data_dir), Path(args.weights), matrix)
    report_path.write_text(report, encoding="utf-8")
    plot_confusion_matrix(matrix, COMPOUND_LABELS_CN, confusion_path)
    save_class_statistics(matrix, statistics_path)

    print(report)
    print(f"RAF-DB Compound Accuracy：{metrics['accuracy']:.4f}")
    print(f"RAF-DB Compound Macro Precision：{metrics['macro_precision']:.4f}")
    print(f"RAF-DB Compound Macro Recall：{metrics['macro_recall']:.4f}")
    print(f"RAF-DB Compound Macro F1-score：{metrics['macro_f1']:.4f}")
    print(f"分类报告已保存：{report_path}")
    print(f"混淆矩阵已保存：{confusion_path}")
    print(f"类别统计已保存：{statistics_path}")
    if args.max_correct_samples > 0:
        print(f"正确样例已保存到：{correct_dir}")
    if args.max_wrong_samples > 0:
        print(f"错误样例已保存到：{wrong_dir}")


if __name__ == "__main__":
    main()
