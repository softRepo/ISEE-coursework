"""RAF-DB 外部泛化评估脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from utils.common import CLASS_NAMES, build_model, ensure_dir, get_device
from utils.dataset import IMAGE_EXTENSIONS, IMAGE_SIZE
from utils.metrics import classification_metrics, confusion_matrix_np
from utils.visualize import plot_confusion_matrix

RAF_TO_FER = {
    1: 5,
    2: 2,
    3: 1,
    4: 3,
    5: 4,
    6: 0,
    7: 6,
}

RAF_CLASS_NAMES = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happy",
    5: "Sad",
    6: "Angry",
    7: "Neutral",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="在 RAF-DB 数据集上测试 FER2013 训练好的模型")
    parser.add_argument("--model", type=str, default="mini_resnet", choices=["simple_cnn", "improved_cnn", "mini_resnet"])
    parser.add_argument("--weights", type=str, default="checkpoints/best_mini_resnet.pth")
    parser.add_argument("--data_dir", type=str, default="data/RAF-DB/DATASET", help="RAF-DB 的 DATASET 目录")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="选择 train 或 test 划分")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_wrong_samples", type=int, default=0, help="最多保存多少个错误分类样例")
    return parser.parse_args()


class RAFDBImageFolderDataset(Dataset):
    """按 ImageFolder 方式读取 RAF-DB 的 1 到 7 类文件夹。"""

    def __init__(self, data_dir: str | Path, split: str = "test"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        self.samples: list[tuple[Path, int, int]] = []

        if not self.split_dir.exists():
            raise FileNotFoundError(f"RAF-DB 划分目录不存在：{self.split_dir}")

        for class_dir in sorted((item for item in self.split_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
            if not class_dir.name.isdigit():
                continue
            raf_label = int(class_dir.name)
            if raf_label not in RAF_TO_FER:
                continue
            fer_label = RAF_TO_FER[raf_label]
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, raf_label, fer_label))

        if not self.samples:
            raise RuntimeError(f"没有在 RAF-DB 目录中找到可用图片：{self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_path, _, fer_label = self.samples[index]
        try:
            image = Image.open(image_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        except Exception as exc:
            raise RuntimeError(f"RAF-DB 图片读取失败：{image_path}") from exc
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor, fer_label, str(image_path)


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
    """将 RAF-DB 灰度 tensor 转为便于保存的图片。"""
    array = image_tensor.squeeze(0).cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L").resize((144, 144), Image.NEAREST).convert("RGB")


def save_wrong_sample(image_tensor: torch.Tensor, true_label: int, pred_label: int, source_path: str, save_path: Path) -> None:
    """保存 RAF-DB 错误分类样例，并标注真实类别与预测类别。"""
    image = tensor_to_pil(image_tensor)
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


def evaluate_rafdb(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    wrong_dir: Path,
    max_wrong_samples: int,
) -> tuple[list[int], list[int]]:
    """在 RAF-DB 数据集上预测，并保存错误分类样例。"""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
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
                if label_int != pred_int and wrong_count < max_wrong_samples:
                    save_wrong_sample(image, label_int, pred_int, image_path, wrong_dir / f"wrong_{wrong_count:03d}.png")
                    wrong_count += 1

    return all_labels, all_preds


def format_rafdb_report(model_name: str, data_dir: Path, split: str, matrix: np.ndarray) -> str:
    """生成 RAF-DB 外部泛化分类报告。"""
    metrics = classification_metrics(matrix)
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    support = matrix.sum(axis=1)

    lines = []
    lines.append("RAF-DB 外部泛化测试分类报告")
    lines.append(f"模型名称：{model_name}")
    lines.append(f"评估划分：{split}")
    lines.append("说明：RAF-DB 原始标签 1 到 7 已按 RAF_TO_FER 映射为 FER2013 的 0 到 6 标签顺序。")
    lines.append("RAF-DB 原始标签顺序：1 Surprise，2 Fear，3 Disgust，4 Happy，5 Sad，6 Angry，7 Neutral。")
    lines.append(f"Accuracy：{metrics['accuracy']:.4f}")
    lines.append(f"Macro Precision：{metrics['macro_precision']:.4f}")
    lines.append(f"Macro Recall：{metrics['macro_recall']:.4f}")
    lines.append(f"Macro F1-score：{metrics['macro_f1']:.4f}")
    lines.append("")
    lines.append(f"{'FER类别':<12}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>10}")
    for index, class_name in enumerate(CLASS_NAMES):
        lines.append(
            f"{class_name:<12}{precision[index]:>12.4f}{recall[index]:>12.4f}{f1[index]:>12.4f}{int(support[index]):>10}"
        )
    return "\n".join(lines)


def save_class_statistics(matrix: np.ndarray, output_path: Path) -> None:
    """保存每个 FER2013 类别在 RAF-DB 上的识别统计。"""
    metrics = classification_metrics(matrix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("RAF-DB 各类别识别效果统计\n")
        file.write("说明：类别顺序为 FER2013 标签顺序，RAF-DB 标签已完成映射。\n")
        for index, class_name in enumerate(CLASS_NAMES):
            total = int(matrix[index].sum())
            correct = int(matrix[index, index])
            file.write(
                f"{class_name}：样本数 {total}，正确 {correct}，"
                f"Precision {metrics['precision'][index]:.4f}，"
                f"Recall {metrics['recall'][index]:.4f}，"
                f"F1 {metrics['f1'][index]:.4f}\n"
            )


def print_mapping() -> None:
    """打印 RAF-DB 到 FER2013 的标签映射，便于检查。"""
    print("RAF-DB 到 FER2013 标签映射：")
    for raf_label in sorted(RAF_TO_FER):
        fer_label = RAF_TO_FER[raf_label]
        print(f"  RAF {raf_label} {RAF_CLASS_NAMES[raf_label]} -> FER {fer_label} {CLASS_NAMES[fer_label]}")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    print("开始加载 RAF-DB 图片文件夹数据集...")
    print_mapping()
    dataset = RAFDBImageFolderDataset(args.data_dir, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"RAF-DB 数据目录：{Path(args.data_dir)}")
    print(f"评估划分：{args.split}")
    print(f"可用样本数：{len(dataset)}")

    device = get_device()
    model = build_model(args.model).to(device)
    load_weights(model, args.weights, device, args.model)
    print(f"模型权重已加载：{args.weights}")
    print(f"当前设备：{device}")

    prefix = f"rafdb_{args.model}"
    wrong_dir = Path(output_dir) / f"{prefix}_wrong_samples"
    all_labels, all_preds = evaluate_rafdb(model, loader, device, wrong_dir, args.max_wrong_samples)
    matrix = confusion_matrix_np(all_labels, all_preds, len(CLASS_NAMES))
    metrics = classification_metrics(matrix)

    report = format_rafdb_report(args.model, Path(args.data_dir), args.split, matrix)
    report_path = Path(output_dir) / f"{prefix}_classification_report.txt"
    confusion_path = Path(output_dir) / f"{prefix}_confusion_matrix.png"
    statistics_path = Path(output_dir) / f"{prefix}_class_statistics.txt"

    report_path.write_text(report, encoding="utf-8")
    plot_confusion_matrix(matrix, CLASS_NAMES, confusion_path)
    save_class_statistics(matrix, statistics_path)

    print(report)
    print(f"RAF-DB Accuracy：{metrics['accuracy']:.4f}")
    print(f"RAF-DB Macro Precision：{metrics['macro_precision']:.4f}")
    print(f"RAF-DB Macro Recall：{metrics['macro_recall']:.4f}")
    print(f"RAF-DB Macro F1-score：{metrics['macro_f1']:.4f}")
    print(f"分类报告已保存：{report_path}")
    print(f"混淆矩阵已保存：{confusion_path}")
    print(f"类别统计已保存：{statistics_path}")
    if args.max_wrong_samples > 0:
        print(f"错误样例已保存到：{wrong_dir}")


if __name__ == "__main__":
    main()
