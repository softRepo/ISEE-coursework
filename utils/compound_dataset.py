"""RAF-DB Compound 复合表情数据集读取与预处理工具。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

COMPOUND_IMAGE_SIZE = 96
COMPOUND_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
COMPOUND_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

COMPOUND_LABELS_EN = [
    "Happily Surprised",
    "Happily Disgusted",
    "Sadly Fearful",
    "Sadly Angry",
    "Sadly Surprised",
    "Sadly Disgusted",
    "Fearfully Angry",
    "Fearfully Surprised",
    "Angrily Surprised",
    "Angrily Disgusted",
    "Disgustedly Surprised",
]

COMPOUND_LABELS_CN = [
    "开心-惊讶",
    "开心-厌恶",
    "悲伤-恐惧",
    "悲伤-愤怒",
    "悲伤-惊讶",
    "悲伤-厌恶",
    "恐惧-愤怒",
    "恐惧-惊讶",
    "愤怒-惊讶",
    "愤怒-厌恶",
    "厌恶-惊讶",
]

COMPOUND_NUM_CLASSES = len(COMPOUND_LABELS_EN)


@dataclass(frozen=True)
class CompoundSample:
    """单个 RAF-DB Compound 样本。"""

    image_path: Path
    label: int
    raw_label: int
    image_name: str


def _find_label_file(data_dir: Path) -> Path:
    """兼容 RAF 官方文件名中的 partition/patition 拼写差异。"""
    candidates = [
        data_dir / "EmoLabel" / "list_partition_label.txt",
        data_dir / "EmoLabel" / "list_patition_label.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "未找到 RAF-DB Compound 标签文件，请确认存在 "
        "data/compound/EmoLabel/list_partition_label.txt 或 list_patition_label.txt。"
    )


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
    raise FileNotFoundError("未找到 RAF-DB Compound aligned 图片目录：data/compound/Image/aligned/")


def _aligned_name(image_name: str) -> str:
    """将标签文件中的原始文件名转换为 aligned 文件名。"""
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

    # 少量数据可能统一使用 jpg 后缀，额外兼容一次。
    jpg_aligned_path = image_dir / _aligned_name(f"{Path(image_name).stem}.jpg")
    if jpg_aligned_path.exists():
        return jpg_aligned_path
    return None


def collect_compound_samples(data_dir: str | Path = "data/compound", split: str = "train") -> list[CompoundSample]:
    """读取 RAF-DB Compound 标签文件，并按 train/test 前缀收集样本。"""
    if split not in {"train", "test"}:
        raise ValueError("split 只能是 train 或 test。")

    root = Path(data_dir)
    label_file = _find_label_file(root)
    image_dir = _find_image_dir(root)
    expected_prefix = f"{split}_"

    samples: list[CompoundSample] = []
    missing_count = 0
    invalid_label_count = 0
    skipped_split_count = 0

    with label_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_name, label_text = parts[0], parts[1]
            if not image_name.startswith(expected_prefix):
                skipped_split_count += 1
                continue
            try:
                raw_label = int(label_text)
            except ValueError:
                invalid_label_count += 1
                continue
            if raw_label < 1 or raw_label > COMPOUND_NUM_CLASSES:
                invalid_label_count += 1
                continue

            image_path = _resolve_image_path(image_dir, image_name)
            if image_path is None:
                missing_count += 1
                continue
            samples.append(
                CompoundSample(
                    image_path=image_path,
                    label=raw_label - 1,
                    raw_label=raw_label,
                    image_name=image_name,
                )
            )

    if not samples:
        raise RuntimeError(
            f"没有在 RAF-DB Compound {split} 集中找到可用样本。"
            f"标签文件：{label_file}，图片目录：{image_dir}"
        )

    print(f"RAF-DB Compound {split} 集读取完成：")
    print(f"  标签文件：{label_file}")
    print(f"  图片目录：{image_dir}")
    print(f"  可用样本：{len(samples)} 张")
    print(f"  跳过其他划分：{skipped_split_count} 行")
    print(f"  缺失图片：{missing_count} 张")
    print(f"  无效标签：{invalid_label_count} 行")
    print_compound_statistics(f"RAF-DB Compound {split} 集", samples)
    return samples


def count_compound_samples(samples: list[CompoundSample]) -> dict[int, int]:
    """统计每个复合表情类别的样本数量。"""
    counter = Counter(sample.label for sample in samples)
    return {index: counter.get(index, 0) for index in range(COMPOUND_NUM_CLASSES)}


def print_compound_statistics(title: str, samples: list[CompoundSample]) -> None:
    """打印复合表情数据集类别分布。"""
    counts = count_compound_samples(samples)
    print(f"{title}：共 {len(samples)} 张")
    for index, class_name in enumerate(COMPOUND_LABELS_CN):
        print(f"  {index} {class_name} / {COMPOUND_LABELS_EN[index]}：{counts[index]} 张")


def _apply_color_jitter(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """对 RGB 图像执行轻量颜色扰动。"""
    if rng.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.85, 1.15)))
    if rng.random() < 0.8:
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.85, 1.15)))
    if rng.random() < 0.8:
        image = ImageEnhance.Color(image).enhance(float(rng.uniform(0.85, 1.15)))
    return image


def preprocess_compound_image(image: Image.Image, use_aug: bool, rng: np.random.Generator) -> torch.Tensor:
    """将 RAF-DB Compound RGB 图片预处理为 [3, 96, 96] tensor。"""
    image = image.convert("RGB").resize((COMPOUND_IMAGE_SIZE, COMPOUND_IMAGE_SIZE), Image.BILINEAR)
    if use_aug:
        if rng.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() < 0.5:
            image = image.rotate(float(rng.uniform(-10.0, 10.0)), resample=Image.BILINEAR)
        image = _apply_color_jitter(image, rng)

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - COMPOUND_MEAN) / COMPOUND_STD
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


class RAFCompoundDataset(Dataset):
    """读取 RAF-DB Compound aligned 图片，用于 11 类复合表情识别。"""

    def __init__(
        self,
        data_dir: str | Path = "data/compound",
        split: str = "train",
        samples: list[CompoundSample] | None = None,
        use_aug: bool = False,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples = samples if samples is not None else collect_compound_samples(self.data_dir, split)
        self.use_aug = use_aug
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[index]
        try:
            image = Image.open(sample.image_path)
        except Exception as exc:
            raise RuntimeError(f"RAF-DB Compound 图片读取失败：{sample.image_path}") from exc
        tensor = preprocess_compound_image(image, self.use_aug, self.rng)
        return tensor, sample.label, str(sample.image_path)


def get_compound_dataloaders(
    data_dir: str | Path = "data/compound",
    batch_size: int = 64,
    num_workers: int = 0,
    use_aug: bool = False,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """返回 RAF-DB Compound 官方 train/test 划分的 DataLoader。"""
    train_samples = collect_compound_samples(data_dir, "train")
    test_samples = collect_compound_samples(data_dir, "test")

    train_dataset = RAFCompoundDataset(data_dir, "train", samples=train_samples, use_aug=use_aug, seed=seed)
    test_dataset = RAFCompoundDataset(data_dir, "test", samples=test_samples, use_aug=False, seed=seed)

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


def unnormalize_compound_tensor(image_tensor: torch.Tensor) -> Image.Image:
    """将归一化后的复合表情 RGB tensor 转回 PIL 图片，便于保存样例。"""
    array = image_tensor.detach().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    array = (array * COMPOUND_STD) + COMPOUND_MEAN
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")
