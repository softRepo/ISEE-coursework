"""RAF-DB RGB 数据集读取与预处理工具。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from utils.common import CLASS_NAMES
from utils.dataset import IMAGE_EXTENSIONS

RGB_IMAGE_SIZE = 96
RGB_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
RGB_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

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


def collect_rafdb_samples(data_dir: str | Path, split: str) -> list[tuple[Path, int, int]]:
    """按 ImageFolder 方式收集 RAF-DB 样本路径和映射后的标签。"""
    root = Path(data_dir)
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"RAF-DB 划分目录不存在：{split_dir}")

    samples: list[tuple[Path, int, int]] = []
    for class_dir in sorted((item for item in split_dir.iterdir() if item.is_dir()), key=lambda item: item.name):
        if not class_dir.name.isdigit():
            continue
        raf_label = int(class_dir.name)
        if raf_label not in RAF_TO_FER:
            continue
        fer_label = RAF_TO_FER[raf_label]
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, raf_label, fer_label))

    if not samples:
        raise RuntimeError(f"没有在 RAF-DB 目录中找到可用图片：{split_dir}")
    return samples


def count_samples_by_fer_label(samples: list[tuple[Path, int, int]]) -> dict[int, int]:
    """统计映射到 FER2013 标签后的每类样本数量。"""
    counts = {label: 0 for label in range(len(CLASS_NAMES))}
    for _, _, fer_label in samples:
        counts[fer_label] += 1
    return counts


def print_sample_statistics(title: str, samples: list[tuple[Path, int, int]]) -> None:
    """打印样本总数和每类数量，便于训练日志和报告记录。"""
    counts = count_samples_by_fer_label(samples)
    print(f"{title}：共 {len(samples)} 张")
    for label, class_name in enumerate(CLASS_NAMES):
        print(f"  {label} {class_name}：{counts[label]} 张")


def stratified_train_val_split(
    samples: list[tuple[Path, int, int]],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[tuple[Path, int, int]], list[tuple[Path, int, int]]]:
    """按类别分层从 RAF-DB train 中划分训练集和验证集。"""
    rng = np.random.default_rng(seed)
    groups: dict[int, list[tuple[Path, int, int]]] = defaultdict(list)
    for sample in samples:
        groups[sample[2]].append(sample)

    train_samples: list[tuple[Path, int, int]] = []
    val_samples: list[tuple[Path, int, int]] = []
    for label in sorted(groups):
        class_samples = groups[label]
        rng.shuffle(class_samples)
        if len(class_samples) <= 1:
            train_samples.extend(class_samples)
            continue
        val_count = max(1, int(round(len(class_samples) * val_ratio)))
        val_count = min(val_count, len(class_samples) - 1)
        val_samples.extend(class_samples[:val_count])
        train_samples.extend(class_samples[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def _random_resized_crop(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """执行轻量随机裁剪并缩放到 96×96。"""
    width, height = image.size
    area = width * height
    for _ in range(8):
        target_area = float(rng.uniform(0.78, 1.0) * area)
        ratio = float(rng.uniform(0.88, 1.12))
        crop_width = int(round((target_area * ratio) ** 0.5))
        crop_height = int(round((target_area / ratio) ** 0.5))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            left = int(rng.integers(0, width - crop_width + 1))
            top = int(rng.integers(0, height - crop_height + 1))
            image = image.crop((left, top, left + crop_width, top + crop_height))
            return image.resize((RGB_IMAGE_SIZE, RGB_IMAGE_SIZE), Image.BILINEAR)
    return image.resize((RGB_IMAGE_SIZE, RGB_IMAGE_SIZE), Image.BILINEAR)


def _apply_color_jitter(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """对 RGB 图像执行亮度、对比度和饱和度扰动。"""
    if rng.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.15)))
    if rng.random() < 0.8:
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.85, 1.15)))
    if rng.random() < 0.8:
        image = ImageEnhance.Color(image).enhance(float(rng.uniform(0.85, 1.15)))
    return image


def preprocess_rgb_image(image: Image.Image, use_aug: bool, rng: np.random.Generator) -> torch.Tensor:
    """将 RAF-DB RGB 图像预处理为 [3, 96, 96] tensor。"""
    image = image.convert("RGB")
    if use_aug:
        image = _random_resized_crop(image, rng)
        if rng.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() < 0.5:
            image = image.rotate(float(rng.uniform(-12.0, 12.0)), resample=Image.BILINEAR)
        image = _apply_color_jitter(image, rng)
    else:
        image = image.resize((RGB_IMAGE_SIZE, RGB_IMAGE_SIZE), Image.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - RGB_MEAN) / RGB_STD
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


class RAFDBRGBDataset(Dataset):
    """读取 RAF-DB RGB 图片并映射到项目统一的 FER2013 标签顺序。"""

    def __init__(
        self,
        data_dir: str | Path = "data/RAF-DB/DATASET",
        split: str = "train",
        samples: list[tuple[Path, int, int]] | None = None,
        use_aug: bool = False,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples = samples if samples is not None else collect_rafdb_samples(self.data_dir, split)
        self.use_aug = use_aug
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_path, _, fer_label = self.samples[index]
        try:
            image = Image.open(image_path)
        except Exception as exc:
            raise RuntimeError(f"RAF-DB 图片读取失败：{image_path}") from exc
        tensor = preprocess_rgb_image(image, self.use_aug, self.rng)
        return tensor, fer_label, str(image_path)


def get_rafdb_rgb_dataloaders(
    data_dir: str | Path = "data/RAF-DB/DATASET",
    batch_size: int = 64,
    num_workers: int = 0,
    use_aug: bool = False,
    seed: int = 42,
    val_ratio: float = 0.15,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """返回 RAF-DB RGB 训练、验证和测试 DataLoader。"""
    train_all_samples = collect_rafdb_samples(data_dir, "train")
    test_samples = collect_rafdb_samples(data_dir, "test")
    train_samples, val_samples = stratified_train_val_split(train_all_samples, val_ratio=val_ratio, seed=seed)

    print_sample_statistics("RAF-DB 原始训练集", train_all_samples)
    print_sample_statistics("RAF-DB 训练子集", train_samples)
    print_sample_statistics("RAF-DB 验证子集", val_samples)
    print_sample_statistics("RAF-DB 测试集", test_samples)

    train_dataset = RAFDBRGBDataset(data_dir, "train", samples=train_samples, use_aug=use_aug, seed=seed)
    val_dataset = RAFDBRGBDataset(data_dir, "train", samples=val_samples, use_aug=False, seed=seed)
    test_dataset = RAFDBRGBDataset(data_dir, "test", samples=test_samples, use_aug=False, seed=seed)

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
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def unnormalize_rgb_tensor(image_tensor: torch.Tensor) -> Image.Image:
    """将归一化后的 RGB tensor 转回 PIL 图片，便于保存样例。"""
    array = image_tensor.detach().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    array = (array * RGB_STD) + RGB_MEAN
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")
