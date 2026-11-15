from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

LABEL_NAME_MAP = {
    0: "T-shirt/top",
    3: "Dress",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
}
TARGET_CLASSES = [0, 3, 5, 6, 7]
ORIG_TO_IDX = {label: idx for idx, label in enumerate(TARGET_CLASSES)}
IDX_TO_ORIG = {idx: label for label, idx in ORIG_TO_IDX.items()}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_dir(split: str) -> Path:
    return project_root() / "data" / split


def default_figures_dir() -> Path:
    return project_root() / "figures"


def default_results_dir() -> Path:
    return project_root() / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_flat_data(data_dir: str | Path, scale_01: bool = False) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "labels.csv")
    images = []
    labels = []
    for _, row in df.iterrows():
        image = Image.open(data_dir / "images" / row["filename"]).convert("L")
        arr = np.asarray(image, dtype=np.float32).reshape(-1)
        if scale_01:
            arr = arr / 255.0
        images.append(arr)
        labels.append(int(row["label_id"]))
    return np.vstack(images), np.asarray(labels, dtype=np.int64)


def class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, float]:
    result = {}
    for label in TARGET_CLASSES:
        mask = y_true == label
        result[label] = float(np.mean(y_pred[mask] == y_true[mask]) * 100.0) if mask.any() else 0.0
    return result


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_table(rows: list[dict], columns: list[str]) -> None:
    widths = {
        col: max(len(col), *(len(f"{row.get(col, '')}") for row in rows))
        for col in columns
    }
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(f"{row.get(col, '')}".ljust(widths[col]) for col in columns))
