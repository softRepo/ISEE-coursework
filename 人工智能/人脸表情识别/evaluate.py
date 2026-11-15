"""FER2013 模型评估脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from utils.common import CLASS_NAMES, build_model, ensure_dir, get_device
from utils.dataset import get_fer2013_dataloaders
from utils.metrics import classification_metrics, confusion_matrix_np, format_classification_report
from utils.visualize import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="评估 FER2013 人脸表情识别模型")
    parser.add_argument("--model", type=str, default="simple_cnn", choices=["simple_cnn", "improved_cnn", "mini_resnet"])
    parser.add_argument("--weights", type=str, default="checkpoints/best_simple_cnn.pth")
    parser.add_argument("--data_dir", type=str, default="data/fer2013")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0, help="最多保存多少个正确/错误样例")
    return parser.parse_args()


def load_weights(model: torch.nn.Module, weights_path: str | Path, device: torch.device) -> None:
    """加载模型权重，并给出清晰中文提示。"""
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"模型权重不存在：{path}，请先运行 train.py 训练模型。")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """将 [1, 48, 48] tensor 转回灰度图片。"""
    array = image_tensor.squeeze(0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L").resize((144, 144), Image.NEAREST).convert("RGB")


def save_sample_image(image_tensor: torch.Tensor, true_label: int, pred_label: int, save_path: Path) -> None:
    """保存带预测说明的样例图片。"""
    image = tensor_to_pil(image_tensor.cpu())
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


def evaluate_and_collect(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    model_name: str,
    max_samples: int,
) -> tuple[list[int], list[int]]:
    """完成测试集预测，并保存部分正确和错误样例。"""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    correct_count = 0
    wrong_count = 0
    correct_dir = output_dir / f"{model_name}_correct_samples"
    wrong_dir = output_dir / f"{model_name}_wrong_samples"

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            for image, label, pred in zip(images, labels, preds):
                label_int = int(label.item())
                pred_int = int(pred.item())
                all_labels.append(label_int)
                all_preds.append(pred_int)
                if label_int == pred_int and correct_count < max_samples:
                    save_sample_image(image, label_int, pred_int, correct_dir / f"correct_{correct_count:03d}.png")
                    correct_count += 1
                elif label_int != pred_int and wrong_count < max_samples:
                    save_sample_image(image, label_int, pred_int, wrong_dir / f"wrong_{wrong_count:03d}.png")
                    wrong_count += 1

    return all_labels, all_preds


def save_class_statistics(matrix: np.ndarray, output_path: Path) -> None:
    """保存每个类别的识别统计。"""
    metrics = classification_metrics(matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("各类别识别效果统计\n")
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
    device = get_device()

    print("开始加载 FER2013 测试集...")
    _, _, test_loader = get_fer2013_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_aug=False,
    )

    model = build_model(args.model).to(device)
    load_weights(model, args.weights, device)
    print(f"模型权重已加载：{args.weights}")
    print(f"当前设备：{device}")

    all_labels, all_preds = evaluate_and_collect(model, test_loader, device, output_dir, args.model, args.max_samples)
    matrix = confusion_matrix_np(all_labels, all_preds, len(CLASS_NAMES))
    metrics = classification_metrics(matrix)
    report = format_classification_report(CLASS_NAMES, matrix)

    report_path = Path(output_dir) / f"{args.model}_classification_report.txt"
    confusion_path = Path(output_dir) / f"{args.model}_confusion_matrix.png"
    statistics_path = Path(output_dir) / f"{args.model}_class_statistics.txt"
    report_path.write_text(report, encoding="utf-8")
    plot_confusion_matrix(matrix, CLASS_NAMES, confusion_path)
    save_class_statistics(matrix, statistics_path)

    print(report)
    print(f"混淆矩阵已保存：{confusion_path}")
    print(f"分类报告已保存：{report_path}")
    print(f"类别统计已保存：{statistics_path}")
    if args.max_samples > 0:
        print(f"正确样例已保存到：{Path(output_dir) / f'{args.model}_correct_samples'}")
    if args.max_samples > 0:
        print(f"错误样例已保存到：{Path(output_dir) / f'{args.model}_wrong_samples'}")
    print(f"总体准确率：{metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
