"""项目公共工具函数。"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

CLASS_NAME_TO_LABEL = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6,
}


def ensure_dir(path: str | os.PathLike) -> Path:
    """确保目录存在，并返回 Path 对象。"""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_seed(seed: int = 42) -> None:
    """固定随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """自动选择可用设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型中需要训练的参数数量。"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_model(model_name: str) -> torch.nn.Module:
    """根据名称创建模型实例。"""
    name = model_name.lower()
    if name == "simple_cnn":
        from models.simple_cnn import SimpleCNN

        return SimpleCNN(num_classes=len(CLASS_NAMES))
    if name == "improved_cnn":
        from models.improved_cnn import ImprovedCNN

        return ImprovedCNN(num_classes=len(CLASS_NAMES))
    if name == "mini_resnet":
        from models.mini_resnet import MiniResNet

        return MiniResNet(num_classes=len(CLASS_NAMES))
    raise ValueError("未知模型名称，请选择 simple_cnn、improved_cnn 或 mini_resnet。")
