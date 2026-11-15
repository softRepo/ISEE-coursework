"""AffectNet YOLO Format RGB 数据集读取工具。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.common import CLASS_NAMES
from utils.dataset import IMAGE_EXTENSIONS
from utils.rafdb_dataset import preprocess_rgb_image, stratified_train_val_split

AFFECTNET_TO_FER = {
    0: 0,
    2: 1,
    3: 2,
    4: 3,
    6: 4,
    7: 5,
    5: 6,
}

AFFECTNET_CLASS_NAMES = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happy",
    5: "Neutral",
    6: "Sad",
    7: "Surprise",
}


def _resolve_base_dir(data_dir: str | Path) -> Path:
    """自动识别 AffectNet 根目录或 YOLO_format 子目录。"""
    root = Path(data_dir)
    if (root / "YOLO_format").exists():
        return root / "YOLO_format"
    return root


def _split_candidates(split: str) -> list[str]:
    """返回常见划分目录候选名称。"""
    if split in {"valid", "validation", "val"}:
        return ["valid", "validation", "val"]
    return [split]


def resolve_yolo_split_dirs(data_dir: str | Path, split: str) -> tuple[Path, Path] | None:
    """自动识别 YOLO 图片目录和标签目录。"""
    base = _resolve_base_dir(data_dir)
    for split_name in _split_candidates(split):
        image_dir = base / split_name / "images"
        label_dir = base / split_name / "labels"
        if image_dir.exists() and label_dir.exists():
            return image_dir, label_dir

        image_dir = base / "images" / split_name
        label_dir = base / "labels" / split_name
        if image_dir.exists() and label_dir.exists():
            return image_dir, label_dir
    return None


def read_first_yolo_label(label_path: Path) -> int | None:
    """读取 YOLO 标签文件第一行的类别 id。"""
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="gbk").splitlines()
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            return int(float(parts[0]))
        except ValueError:
            return None
    return None


def collect_affectnet_samples(
    data_dir: str | Path,
    split: str = "train",
    max_per_class: int | None = 1000,
    use_all_data: bool = False,
) -> list[tuple[Path, int, int]]:
    """收集 AffectNet YOLO 样本，并映射到 FER2013 标签顺序。"""
    resolved = resolve_yolo_split_dirs(data_dir, split)
    if resolved is None:
        root = _resolve_base_dir(data_dir)
        print(f"没有自动识别到 AffectNet {split} 的 images/labels 目录。")
        print("请整理为以下任一结构：")
        print("  data/AffectNet/YOLO_format/train/images 与 train/labels")
        print("  data/AffectNet/images/train 与 labels/train")
        raise FileNotFoundError(f"AffectNet {split} 目录结构不符合预期：{root}")

    image_dir, label_dir = resolved
    effective_limit = 0 if use_all_data or max_per_class in (None, 0) else int(max_per_class)
    counts = {label: 0 for label in range(len(CLASS_NAMES))}
    samples: list[tuple[Path, int, int]] = []
    skipped_contempt = 0
    skipped_missing_label = 0
    skipped_invalid_label = 0

    image_paths = sorted(
        image_path
        for image_path in image_dir.rglob("*")
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for image_path in image_paths:
        relative_path = image_path.relative_to(image_dir)
        label_path = (label_dir / relative_path).with_suffix(".txt")
        if not label_path.exists():
            skipped_missing_label += 1
            continue
        affect_label = read_first_yolo_label(label_path)
        if affect_label == 1:
            skipped_contempt += 1
            continue
        if affect_label not in AFFECTNET_TO_FER:
            skipped_invalid_label += 1
            continue
        fer_label = AFFECTNET_TO_FER[affect_label]
        if effective_limit > 0 and counts[fer_label] >= effective_limit:
            continue
        samples.append((image_path, affect_label, fer_label))
        counts[fer_label] += 1
        if effective_limit > 0 and all(count >= effective_limit for count in counts.values()):
            break

    if not samples:
        raise RuntimeError(f"AffectNet {split} 没有可用样本，请检查图片和标签文件。")

    limit_text = "读取全部有效样本" if effective_limit <= 0 else f"每个 FER 类别最多读取 {effective_limit} 张"
    print(f"AffectNet 当前 split：{split}")
    print(f"AffectNet {split} 图片目录：{image_dir}")
    print(f"AffectNet {split} 标签目录：{label_dir}")
    print(f"AffectNet {split} 读取策略：{limit_text}")
    print(f"跳过 Contempt 样本：{skipped_contempt}，缺少标签：{skipped_missing_label}，无效标签：{skipped_invalid_label}")
    return samples


def print_affectnet_statistics(title: str, samples: list[tuple[Path, int, int]]) -> None:
    """打印 AffectNet 映射后每类样本数量。"""
    counts = {label: 0 for label in range(len(CLASS_NAMES))}
    raw_counts: dict[int, int] = defaultdict(int)
    for _, affect_label, fer_label in samples:
        counts[fer_label] += 1
        raw_counts[affect_label] += 1

    print(f"{title}：共 {len(samples)} 张")
    for label, class_name in enumerate(CLASS_NAMES):
        print(f"  FER {label} {class_name}：{counts[label]} 张")
    print("AffectNet 原始类别计数：")
    for affect_label in sorted(raw_counts):
        print(f"  AffectNet {affect_label} {AFFECTNET_CLASS_NAMES.get(affect_label, 'Unknown')}：{raw_counts[affect_label]} 张")


class AffectNetYOLODataset(Dataset):
    """读取 AffectNet YOLO Format 图像并返回 RGB tensor。"""

    def __init__(
        self,
        samples: list[tuple[Path, int, int]],
        use_aug: bool = False,
        seed: int = 42,
    ):
        self.samples = samples
        self.use_aug = use_aug
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_path, _, fer_label = self.samples[index]
        try:
            from PIL import Image

            image = Image.open(image_path)
        except Exception as exc:
            raise RuntimeError(f"AffectNet 图片读取失败：{image_path}") from exc
        tensor = preprocess_rgb_image(image, self.use_aug, self.rng)
        return tensor, fer_label, str(image_path)


def get_affectnet_rgb_dataloaders(
    data_dir: str | Path = "data/AffectNet/YOLO_format",
    batch_size: int = 128,
    num_workers: int = 0,
    max_per_class: int | None = 1000,
    use_all_data: bool = False,
    use_aug: bool = False,
    seed: int = 42,
    val_ratio: float = 0.10,
) -> tuple[DataLoader, DataLoader]:
    """返回 AffectNet 训练和验证 DataLoader。"""
    train_samples_all = collect_affectnet_samples(
        data_dir,
        "train",
        max_per_class=max_per_class,
        use_all_data=use_all_data,
    )
    valid_samples = None
    for valid_name in ["valid", "validation", "val"]:
        if resolve_yolo_split_dirs(data_dir, valid_name) is not None:
            valid_samples = collect_affectnet_samples(
                data_dir,
                valid_name,
                max_per_class=max_per_class,
                use_all_data=use_all_data,
            )
            break

    if valid_samples is None:
        train_samples, valid_samples = stratified_train_val_split(train_samples_all, val_ratio=val_ratio, seed=seed)
        print("未找到 AffectNet 验证集，已从训练集中按类别划分验证集。")
    else:
        train_samples = train_samples_all

    print_affectnet_statistics("AffectNet 训练集", train_samples)
    print_affectnet_statistics("AffectNet 验证集", valid_samples)

    train_dataset = AffectNetYOLODataset(train_samples, use_aug=use_aug, seed=seed)
    valid_dataset = AffectNetYOLODataset(valid_samples, use_aug=False, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader
