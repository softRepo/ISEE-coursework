"""FER2013 与 CK+ 数据集读取模块。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils.common import CLASS_NAME_TO_LABEL, CLASS_NAMES

IMAGE_SIZE = 48
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _folder_to_label(folder_name: str) -> int | None:
    """把数字或英文类别文件夹名统一转换为 0 到 6 的标签。"""
    name = folder_name.strip().lower()
    if name.isdigit():
        label = int(name)
        return label if 0 <= label < len(CLASS_NAMES) else None
    return CLASS_NAME_TO_LABEL.get(name)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """将 PIL 灰度图转换为 [1, 48, 48] 的归一化 tensor。"""
    image = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _augment_image(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """执行简单数据增强，保持实现轻量且适合课程项目。"""
    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        angle = float(rng.uniform(-12.0, 12.0))
        image = image.rotate(angle, resample=Image.BILINEAR)
    return image


class FER2013ImageDataset(Dataset):
    """读取 FER2013 图片文件夹数据集。"""

    def __init__(self, split_dir: str | Path, use_aug: bool = False, seed: int = 42):
        self.split_dir = Path(split_dir)
        self.use_aug = use_aug
        self.rng = np.random.default_rng(seed)
        self.samples: list[tuple[Path, int]] = []

        if not self.split_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在：{self.split_dir}")

        for class_dir in sorted(item for item in self.split_dir.iterdir() if item.is_dir()):
            label = _folder_to_label(class_dir.name)
            if label is None:
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise RuntimeError(f"没有在目录中找到可用图片：{self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        try:
            image = Image.open(image_path).convert("L")
        except Exception as exc:
            raise RuntimeError(f"图片读取失败：{image_path}") from exc
        if self.use_aug:
            image = _augment_image(image, self.rng)
        return _pil_to_tensor(image), label


class CKPlusTableDataset(Dataset):
    """读取 CK+ Excel 或 CSV 表格数据集。"""

    def __init__(self, file_path: str | Path, usage: str | None = None):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"CK+ 表格文件不存在：{self.file_path}")

        suffix = self.file_path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(self.file_path)
        elif suffix == ".csv":
            frame = pd.read_csv(self.file_path)
        else:
            raise ValueError("CK+ 文件后缀必须是 .xlsx、.xls 或 .csv。")

        frame.columns = [str(column).strip().lower() for column in frame.columns]
        required_columns = {"emotion", "pixels", "usage"}
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(f"CK+ 表格缺少必要列：{sorted(missing_columns)}")

        frame = frame[frame["emotion"].astype(int) != 7].copy()
        if usage is not None:
            usage_lower = usage.lower()
            frame = frame[frame["usage"].astype(str).str.lower() == usage_lower].copy()

        if frame.empty:
            raise RuntimeError("CK+ 表格筛选后没有可用样本，请检查 Usage 或标签。")

        self.labels = frame["emotion"].astype(int).to_numpy()
        self.pixels = frame["pixels"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        values = np.fromstring(self.pixels[index], sep=" ", dtype=np.float32)
        if values.size != IMAGE_SIZE * IMAGE_SIZE:
            raise RuntimeError(f"第 {index} 个 CK+ 样本像素数量不是 2304。")
        image = values.reshape(IMAGE_SIZE, IMAGE_SIZE) / 255.0
        tensor = torch.from_numpy(image).unsqueeze(0)
        return tensor, int(self.labels[index])


def _resolve_split_dir(data_dir: Path, candidates: list[str]) -> Path:
    """根据候选名称查找数据划分目录。"""
    for name in candidates:
        split_dir = data_dir / name
        if split_dir.exists():
            return split_dir
    names = "、".join(candidates)
    raise FileNotFoundError(f"没有找到 {names} 数据目录，请检查路径：{data_dir}")


def get_fer2013_datasets(
    data_dir: str | Path = "data/fer2013",
    use_aug: bool = False,
    seed: int = 42,
) -> tuple[FER2013ImageDataset, FER2013ImageDataset, FER2013ImageDataset]:
    """返回 FER2013 的 train、validation、test 三个 Dataset。"""
    root = Path(data_dir)
    train_dir = _resolve_split_dir(root, ["train"])
    val_dir = _resolve_split_dir(root, ["validation", "val"])
    test_dir = _resolve_split_dir(root, ["test"])

    train_dataset = FER2013ImageDataset(train_dir, use_aug=use_aug, seed=seed)
    val_dataset = FER2013ImageDataset(val_dir, use_aug=False, seed=seed)
    test_dataset = FER2013ImageDataset(test_dir, use_aug=False, seed=seed)
    return train_dataset, val_dataset, test_dataset


def get_fer2013_dataloaders(
    data_dir: str | Path = "data/fer2013",
    batch_size: int = 64,
    num_workers: int = 0,
    use_aug: bool = False,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """返回 FER2013 的 train、validation、test 三个 DataLoader。"""
    train_dataset, val_dataset, test_dataset = get_fer2013_datasets(data_dir, use_aug, seed)
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


def describe_fer2013(data_dir: str | Path = "data/fer2013") -> None:
    """打印 FER2013 数据目录样本数量，便于命令行检查。"""
    root = Path(data_dir)
    split_map = {
        "train": _resolve_split_dir(root, ["train"]),
        "validation": _resolve_split_dir(root, ["validation", "val"]),
        "test": _resolve_split_dir(root, ["test"]),
    }
    print("FER2013 数据集结构检查")
    print(f"数据根目录：{root}")
    for split_name, split_dir in split_map.items():
        print(f"\n{split_name}：{split_dir}")
        total = 0
        for class_dir in sorted(item for item in split_dir.iterdir() if item.is_dir()):
            label = _folder_to_label(class_dir.name)
            if label is None:
                continue
            count = sum(
                1
                for image_path in class_dir.rglob("*")
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
            )
            total += count
            print(f"  {class_dir.name} -> {CLASS_NAMES[label]}：{count} 张")
        print(f"  合计：{total} 张")


def main() -> None:
    """提供一个数据结构检查入口。"""
    parser = argparse.ArgumentParser(description="检查 FER2013 数据集结构")
    parser.add_argument("--data_dir", type=str, default="data/fer2013", help="FER2013 数据集根目录")
    args = parser.parse_args()
    describe_fer2013(args.data_dir)


if __name__ == "__main__":
    main()
