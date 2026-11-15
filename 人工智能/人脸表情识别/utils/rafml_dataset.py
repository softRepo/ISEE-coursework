"""RAF-ML 多标签混合表情数据集读取与预处理工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

RAFML_IMAGE_SIZE = 96
RAFML_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
RAFML_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

RAFML_LABELS_EN = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
]

RAFML_LABELS_CN = [
    "惊讶",
    "恐惧",
    "厌恶",
    "开心",
    "悲伤",
    "愤怒",
]

RAFML_NUM_LABELS = len(RAFML_LABELS_EN)


@dataclass(frozen=True)
class RAFMLSample:
    """单个 RAF-ML 多标签样本。"""

    image_path: Path
    labels: tuple[float, ...]
    image_name: str
    partition: int


def _has_images(directory: Path) -> bool:
    """判断目录下是否直接包含 jpg/jpeg/png 图片。"""
    if not directory.exists() or not directory.is_dir():
        return False
    return any(item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"} for item in directory.iterdir())


def _find_image_dir(data_dir: Path) -> Path:
    """兼容 Image/aligned 和 Image/aligned/aligned 两种目录层级。"""
    candidates = [
        data_dir / "Image" / "aligned",
        data_dir / "Image" / "aligned" / "aligned",
    ]
    for candidate in candidates:
        if _has_images(candidate):
            return candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("未找到 RAF-ML aligned 图片目录：data/RAF-ML/Image/aligned/")


def _aligned_name(image_name: str) -> str:
    """将原始文件名转换为 aligned 文件名。"""
    path = Path(image_name)
    stem = path.stem
    suffix = path.suffix or ".jpg"
    if stem.endswith("_aligned"):
        return f"{stem}{suffix}"
    return f"{stem}_aligned{suffix}"


def _resolve_image_path(image_dir: Path, image_name: str) -> Path | None:
    """按直接文件名和 aligned 文件名依次查找图片。"""
    direct_path = image_dir / image_name
    if direct_path.exists():
        return direct_path

    aligned_path = image_dir / _aligned_name(image_name)
    if aligned_path.exists():
        return aligned_path

    jpg_aligned_path = image_dir / _aligned_name(f"{Path(image_name).stem}.jpg")
    if jpg_aligned_path.exists():
        return jpg_aligned_path
    return None


def _read_partition_file(partition_path: Path) -> dict[str, int]:
    """读取 RAF-ML 官方 train/test 划分文件。"""
    if not partition_path.exists():
        raise FileNotFoundError(f"未找到 RAF-ML 划分文件：{partition_path}")

    partition_map: dict[str, int] = {}
    invalid_count = 0
    with partition_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_name, partition_text = parts[0], parts[1]
            try:
                partition = int(partition_text)
            except ValueError:
                invalid_count += 1
                continue
            if partition not in {0, 1}:
                invalid_count += 1
                continue
            partition_map[image_name] = partition
    if invalid_count:
        print(f"RAF-ML 划分文件中发现 {invalid_count} 行无效划分，已跳过。")
    return partition_map


def _read_multilabel_file(multilabel_path: Path) -> tuple[dict[str, tuple[float, ...]], int]:
    """读取 RAF-ML 六维多标签文件。"""
    if not multilabel_path.exists():
        raise FileNotFoundError(f"未找到 RAF-ML 多标签文件：{multilabel_path}")

    label_map: dict[str, tuple[float, ...]] = {}
    invalid_count = 0
    with multilabel_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < RAFML_NUM_LABELS + 1:
                invalid_count += 1
                continue
            image_name = parts[0]
            try:
                labels = tuple(float(value) for value in parts[1 : 1 + RAFML_NUM_LABELS])
            except ValueError:
                invalid_count += 1
                continue
            if any(label not in {0.0, 1.0} for label in labels):
                invalid_count += 1
                continue
            label_map[image_name] = labels
    return label_map, invalid_count


def collect_rafml_samples(data_dir: str | Path = "data/RAF-ML", split: str = "train") -> list[RAFMLSample]:
    """读取 RAF-ML 多标签样本，并按官方 train/test 划分过滤。"""
    if split not in {"train", "test"}:
        raise ValueError("split 只能是 train 或 test。")

    root = Path(data_dir)
    image_dir = _find_image_dir(root)
    multilabel_path = root / "EmoLabel" / "multilabel.txt"
    partition_path = root / "EmoLabel" / "partition_label.txt"
    label_map, invalid_label_count = _read_multilabel_file(multilabel_path)
    partition_map = _read_partition_file(partition_path)
    target_partition = 0 if split == "train" else 1

    samples: list[RAFMLSample] = []
    missing_image_count = 0
    missing_partition_count = 0
    skipped_split_count = 0

    for image_name, labels in label_map.items():
        if image_name not in partition_map:
            missing_partition_count += 1
            continue
        partition = partition_map[image_name]
        if partition != target_partition:
            skipped_split_count += 1
            continue
        image_path = _resolve_image_path(image_dir, image_name)
        if image_path is None:
            missing_image_count += 1
            continue
        samples.append(RAFMLSample(image_path=image_path, labels=labels, image_name=image_name, partition=partition))

    if not samples:
        raise RuntimeError(f"没有在 RAF-ML {split} 集中找到可用样本。")

    print(f"RAF-ML {split} 集读取完成：")
    print(f"  多标签文件：{multilabel_path}")
    print(f"  划分文件：{partition_path}")
    print(f"  图片目录：{image_dir}")
    print(f"  可用样本：{len(samples)} 张")
    print(f"  跳过其他划分：{skipped_split_count} 行")
    print(f"  缺失划分：{missing_partition_count} 行")
    print(f"  缺失图片：{missing_image_count} 张")
    print(f"  无效标签：{invalid_label_count} 行")
    print_rafml_statistics(f"RAF-ML {split} 集", samples)
    return samples


def count_positive_labels(samples: list[RAFMLSample]) -> np.ndarray:
    """统计每个标签维度的正样本数量。"""
    counts = np.zeros(RAFML_NUM_LABELS, dtype=np.int64)
    for sample in samples:
        counts += np.asarray(sample.labels, dtype=np.int64)
    return counts


def print_rafml_statistics(title: str, samples: list[RAFMLSample]) -> None:
    """打印 RAF-ML 多标签样本数量和每类正样本数量。"""
    counts = count_positive_labels(samples)
    print(f"{title}：共 {len(samples)} 张")
    for index, count in enumerate(counts.tolist()):
        print(f"  {index} {RAFML_LABELS_CN[index]} / {RAFML_LABELS_EN[index]}：正样本 {count} 张")


def _apply_color_jitter(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """对 RGB 图像执行轻量颜色扰动。"""
    if rng.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.15)))
    if rng.random() < 0.8:
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.85, 1.15)))
    if rng.random() < 0.8:
        image = ImageEnhance.Color(image).enhance(float(rng.uniform(0.85, 1.15)))
    return image


def preprocess_rafml_image(image: Image.Image, use_aug: bool, rng: np.random.Generator) -> torch.Tensor:
    """将 RAF-ML RGB 图片预处理为 [3, 96, 96] tensor。"""
    image = image.convert("RGB").resize((RAFML_IMAGE_SIZE, RAFML_IMAGE_SIZE), Image.BILINEAR)
    if use_aug:
        if rng.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() < 0.5:
            image = image.rotate(float(rng.uniform(-10.0, 10.0)), resample=Image.BILINEAR)
        image = _apply_color_jitter(image, rng)

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - RAFML_MEAN) / RAFML_STD
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


class RAFMLDataset(Dataset):
    """读取 RAF-ML aligned 图片，用于 6 维 sigmoid 多标签识别。"""

    def __init__(
        self,
        data_dir: str | Path = "data/RAF-ML",
        split: str = "train",
        transform=None,
        label_mode: str = "multilabel",
        samples: list[RAFMLSample] | None = None,
        use_aug: bool = False,
        seed: int = 42,
    ):
        if label_mode != "multilabel":
            raise ValueError("RAFMLDataset 当前仅支持 label_mode='multilabel'。")
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.label_mode = label_mode
        self.samples = samples if samples is not None else collect_rafml_samples(self.data_dir, split)
        self.use_aug = use_aug
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[index]
        try:
            image = Image.open(sample.image_path)
        except Exception as exc:
            raise RuntimeError(f"RAF-ML 图片读取失败：{sample.image_path}") from exc
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = preprocess_rafml_image(image, self.use_aug, self.rng)
        label_tensor = torch.tensor(sample.labels, dtype=torch.float32)
        return image_tensor, label_tensor, str(sample.image_path)


def get_rafml_dataloaders(
    data_dir: str | Path = "data/RAF-ML",
    batch_size: int = 64,
    num_workers: int = 0,
    use_aug: bool = False,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """返回 RAF-ML 官方 train/test 划分的 DataLoader。"""
    train_samples = collect_rafml_samples(data_dir, "train")
    test_samples = collect_rafml_samples(data_dir, "test")

    train_dataset = RAFMLDataset(data_dir, "train", samples=train_samples, use_aug=use_aug, seed=seed)
    test_dataset = RAFMLDataset(data_dir, "test", samples=test_samples, use_aug=False, seed=seed)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def unnormalize_rafml_tensor(image_tensor: torch.Tensor) -> Image.Image:
    """将归一化后的 RAF-ML RGB tensor 转回 PIL 图片，便于保存样例。"""
    array = image_tensor.detach().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    array = (array * RAFML_STD) + RAFML_MEAN
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")
