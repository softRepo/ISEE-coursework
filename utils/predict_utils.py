"""统一图片表情预测工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from models.mini_resnet import MiniResNet
from models.rgb_mini_resnet import RGBMiniResNet
from utils.common import CLASS_NAMES
from utils.rafdb_dataset import RGB_MEAN, RGB_STD

CLASS_NAMES_CN = ["愤怒", "厌恶", "恐惧", "开心", "悲伤", "惊讶", "中性"]


def load_emotion_model(
    model_name: str,
    weights_path: str | Path,
    device: torch.device | str | None = None,
    num_classes: int | None = None,
) -> torch.nn.Module:
    """根据模型名称、类别数和权重路径加载表情识别模型。"""
    name = model_name.strip().lower()
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"模型权重文件不存在，请检查路径：{path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    class_count = int(num_classes) if num_classes is not None else len(CLASS_NAMES)
    if class_count <= 0:
        raise ValueError("num_classes 必须是正整数。")

    if name == "mini_resnet":
        model = MiniResNet(num_classes=class_count)
    elif name == "rgb_mini_resnet":
        model = RGBMiniResNet(num_classes=class_count)
    else:
        raise ValueError("不支持的模型类型，请选择 mini_resnet 或 rgb_mini_resnet。")

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"模型权重与当前模型结构或类别数不匹配，请确认 num_classes={class_count}，权重路径：{path}"
        ) from exc
    model.to(device)
    model.eval()
    return model


def _looks_like_bgr(array: np.ndarray) -> bool:
    """根据人脸图像常见红蓝通道强度，粗略判断 OpenCV BGR 输入。"""
    if array.ndim != 3 or array.shape[2] < 3:
        return False
    channel_0_mean = float(array[:, :, 0].mean())
    channel_2_mean = float(array[:, :, 2].mean())
    return channel_2_mean > channel_0_mean * 1.08


def _open_image(image_source: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """读取图片路径、PIL 图片或 RGB numpy 图像。"""
    if isinstance(image_source, Image.Image):
        return image_source.copy()
    if isinstance(image_source, np.ndarray):
        array = np.asarray(image_source)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            return Image.fromarray(array, mode="L").convert("RGB")
        if array.ndim == 3 and array.shape[2] >= 3:
            rgb_array = array[:, :, :3]
            # GUI 摄像头帧已统一保存为 RGB；若外部传入疑似 OpenCV BGR，则自动转为 RGB。
            if _looks_like_bgr(rgb_array):
                rgb_array = rgb_array[:, :, ::-1]
            rgb_array = np.ascontiguousarray(rgb_array)
            return Image.fromarray(rgb_array, mode="RGB")
        raise ValueError("numpy 图像格式不支持，请输入 H×W、H×W×3 或 H×W×4 数组。")
    image_path = Path(image_source)
    if not image_path.exists():
        raise FileNotFoundError(f"图片文件不存在：{image_path}")
    return Image.open(image_path)


def preprocess_image_for_model(
    image_source: str | Path | Image.Image | np.ndarray,
    model_name: str,
) -> tuple[torch.Tensor, Image.Image]:
    """根据模型类型完成图片预处理，并返回模型输入 tensor 和显示原图。"""
    name = model_name.strip().lower()
    original_image = _open_image(image_source).convert("RGB")

    if name == "mini_resnet":
        image = original_image.convert("L").resize((48, 48), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    elif name == "rgb_mini_resnet":
        image = original_image.convert("RGB").resize((96, 96), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - RGB_MEAN) / RGB_STD
        array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(array).unsqueeze(0)
    else:
        raise ValueError("不支持的模型类型，请选择 mini_resnet 或 rgb_mini_resnet。")

    return tensor.float(), original_image


def predict_emotion(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """输入模型和预处理 tensor，返回预测类别、置信度和七类概率。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    predicted_index = int(np.argmax(probs))
    probability_list = [float(prob * 100.0) for prob in probs]
    probability_dict = {
        CLASS_NAMES[index]: probability_list[index]
        for index in range(len(CLASS_NAMES))
    }
    confidence = probability_list[predicted_index]

    return {
        "predicted_index": predicted_index,
        "predicted_label_en": CLASS_NAMES[predicted_index],
        "predicted_label_cn": CLASS_NAMES_CN[predicted_index],
        "confidence": confidence,
        "probability_dict": probability_dict,
        "probability_list": probability_list,
    }


def predict_image(
    image: str | Path | Image.Image | np.ndarray,
    model: torch.nn.Module,
    model_name: str,
    task_type: str,
    output_type: str,
    labels_cn: list[str],
    labels_en: list[str],
    device: torch.device | str | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """通用图片预测函数，支持七类、复合表情和多标签混合表情任务。"""
    if len(labels_cn) != len(labels_en):
        raise ValueError("中英文标签数量不一致，请检查 GUI 模型配置。")
    if not labels_cn:
        raise ValueError("标签列表不能为空。")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    input_tensor, _ = preprocess_image_for_model(image, model_name)
    output_name = output_type.strip().lower()

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        if logits.shape[1] != len(labels_cn):
            raise ValueError(
                f"模型输出维度为 {logits.shape[1]}，但当前任务标签数为 {len(labels_cn)}，请检查权重和模型选项。"
            )
        if output_name == "softmax":
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        elif output_name == "sigmoid":
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        else:
            raise ValueError("不支持的输出类型，请选择 softmax 或 sigmoid。")

    probability_list = [float(prob * 100.0) for prob in probs]
    predicted_index = int(np.argmax(probs))
    confidence = probability_list[predicted_index]
    probability_dict = {
        labels_en[index]: probability_list[index]
        for index in range(len(labels_en))
    }

    result: dict[str, Any] = {
        "task_type": task_type,
        "output_type": output_name,
        "threshold": float(threshold),
        "probs": [float(prob) for prob in probs],
        "predicted_index": predicted_index,
        "predicted_label_en": labels_en[predicted_index],
        "predicted_label_cn": labels_cn[predicted_index],
        "confidence": confidence,
        "top_prob": confidence,
        "probability_dict": probability_dict,
        "probability_list": probability_list,
    }

    if output_name == "sigmoid":
        threshold_percent = float(threshold) * 100.0
        result["active_labels"] = [
            {
                "index": index,
                "label_cn": labels_cn[index],
                "label_en": labels_en[index],
                "probability": probability,
            }
            for index, probability in enumerate(probability_list)
            if probability >= threshold_percent
        ]

    return result
