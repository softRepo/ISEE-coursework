"""RAF-ML 多标签混合表情 RGB 模型训练脚本。"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.rgb_mini_resnet import RGBMiniResNet
from utils.common import count_parameters, ensure_dir, get_device, set_seed
from utils.rafml_dataset import RAFML_LABELS_CN, RAFML_LABELS_EN, RAFML_NUM_LABELS, get_rafml_dataloaders
from utils.visualize import setup_chinese_font

DEFAULT_TAG = "rafml_rgb_mini_resnet"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练 RAF-ML 6 维多标签混合表情 RGB 模型")
    parser.add_argument("--data_dir", type=str, default="data/RAF-ML")
    parser.add_argument("--model", type=str, default="rafml_rgb_mini_resnet", choices=["rafml_rgb_mini_resnet", "rgb_mini_resnet"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--use_aug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG, help="输出文件名前缀，用于避免覆盖其他实验")
    parser.add_argument("--pretrained_weights", type=str, default="", help="可选 RGB 预训练权重路径，只加载特征层，跳过分类层")
    parser.add_argument("--threshold", type=float, default=0.5, help="sigmoid 概率转二值标签的阈值")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    return parser.parse_args()


def build_rafml_model(model_name: str) -> nn.Module:
    """创建 6 维多标签 RGB MiniResNet。"""
    if model_name in {"rafml_rgb_mini_resnet", "rgb_mini_resnet"}:
        return RGBMiniResNet(num_classes=RAFML_NUM_LABELS)
    raise ValueError("未知模型名称，目前仅支持 rafml_rgb_mini_resnet 或 rgb_mini_resnet。")


def _clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """兼容 DataParallel 保存的 module. 前缀。"""
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def load_backbone_pretrained_weights(model: nn.Module, weights_path: str | Path, device: torch.device) -> None:
    """加载 RGB 预训练权重中的特征层，跳过最后分类层。"""
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"预训练权重不存在：{path}")

    checkpoint = torch.load(path, map_location=device)
    raw_state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    pretrained_state = _clean_state_dict_keys(raw_state_dict)
    current_state = model.state_dict()

    loadable_state = {}
    skipped_keys: list[str] = []
    for key, value in pretrained_state.items():
        if key.startswith("classifier."):
            skipped_keys.append(key)
            continue
        if key in current_state and current_state[key].shape == value.shape:
            loadable_state[key] = value
        else:
            skipped_keys.append(key)

    current_state.update(loadable_state)
    model.load_state_dict(current_state)
    print(f"预训练微调模式：已加载 RGB 预训练特征层权重：{path}")
    print(f"  成功加载参数：{len(loadable_state)} 个")
    print(f"  跳过分类层或形状不匹配参数：{len(skipped_keys)} 个")


def multilabel_metrics_from_tensors(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """根据 sigmoid 概率和阈值计算多标签 micro/macro 指标。"""
    preds = (probabilities >= threshold).to(torch.int64)
    targets = labels.to(torch.int64)
    true_positive = (preds * targets).sum(dim=0).to(torch.float64)
    predicted_positive = preds.sum(dim=0).to(torch.float64)
    actual_positive = targets.sum(dim=0).to(torch.float64)
    eps = 1e-12

    precision = true_positive / torch.clamp(predicted_positive, min=eps)
    recall = true_positive / torch.clamp(actual_positive, min=eps)
    f1 = 2 * precision * recall / torch.clamp(precision + recall, min=eps)

    micro_tp = true_positive.sum()
    micro_pred = predicted_positive.sum()
    micro_actual = actual_positive.sum()
    micro_precision = micro_tp / torch.clamp(micro_pred, min=eps)
    micro_recall = micro_tp / torch.clamp(micro_actual, min=eps)
    micro_f1 = 2 * micro_precision * micro_recall / torch.clamp(micro_precision + micro_recall, min=eps)

    return {
        "micro_precision": float(micro_precision.item()),
        "micro_recall": float(micro_recall.item()),
        "micro_f1": float(micro_f1.item()),
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
    }


def run_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    """执行一个训练或测试轮次，并返回 loss、micro F1、macro F1。"""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total = 0
    all_labels: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size
        all_labels.append(labels.detach().cpu())
        all_probs.append(torch.sigmoid(logits.detach()).cpu())

    labels_tensor = torch.cat(all_labels, dim=0)
    probs_tensor = torch.cat(all_probs, dim=0)
    metrics = multilabel_metrics_from_tensors(labels_tensor, probs_tensor, threshold)
    return total_loss / max(total, 1), metrics["micro_f1"], metrics["macro_f1"]


def evaluate_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    """在给定数据集上计算 RAF-ML 多标签总体指标。"""
    model.eval()
    all_labels: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels, _ in loader:
            logits = model(images.to(device))
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())
    return multilabel_metrics_from_tensors(torch.cat(all_labels, dim=0), torch.cat(all_probs, dim=0), threshold)


def save_train_log(history: list[dict[str, float]], log_path: Path) -> None:
    """保存 RAF-ML 训练日志为 CSV 文件。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_micro_f1", "train_macro_f1", "val_loss", "val_micro_f1", "val_macro_f1", "lr", "epoch_time"],
        )
        writer.writeheader()
        writer.writerows(history)


def plot_rafml_training_curves(history: list[dict[str, float]], output_dir: str | Path, tag: str) -> None:
    """保存 RAF-ML loss 曲线和 F1 曲线。"""
    setup_chinese_font()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="训练损失")
    plt.plot(epochs, [row["val_loss"] for row in history], label="测试损失")
    plt.xlabel("轮次")
    plt.ylabel("损失")
    plt.title(f"{tag} 损失曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"{tag}_loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_micro_f1"] for row in history], label="训练 Micro F1")
    plt.plot(epochs, [row["train_macro_f1"] for row in history], label="训练 Macro F1")
    plt.plot(epochs, [row["val_micro_f1"] for row in history], label="测试 Micro F1")
    plt.plot(epochs, [row["val_macro_f1"] for row in history], label="测试 Macro F1")
    plt.xlabel("轮次")
    plt.ylabel("F1-score")
    plt.title(f"{tag} F1 曲线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"{tag}_f1_curve.png", dpi=160)
    plt.close()


def append_experiment_result(result_path: Path, row: dict[str, object]) -> None:
    """将 RAF-ML 实验摘要追加到统一实验结果表。"""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "use_aug",
        "epochs",
        "batch_size",
        "lr",
        "best_epoch",
        "best_val_acc",
        "test_acc",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "parameters",
        "total_time_sec",
    ]
    need_header = not result_path.exists()
    with result_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """训练 RAF-ML 多标签混合表情 RGB 模型。"""
    args = parse_args()
    set_seed(args.seed)

    save_dir = ensure_dir(args.save_dir)
    log_dir = ensure_dir(args.log_dir)
    output_dir = ensure_dir(args.output_dir)
    tag = args.tag.strip() or DEFAULT_TAG
    threshold = max(0.0, min(float(args.threshold), 1.0))

    print("开始加载 RAF-ML 多标签混合表情数据集...")
    train_loader, test_loader = get_rafml_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_aug=args.use_aug,
        seed=args.seed,
    )

    device = get_device()
    model = build_rafml_model(args.model).to(device)
    training_mode = "预训练微调" if args.pretrained_weights.strip() else "从头训练"
    if args.pretrained_weights.strip():
        load_backbone_pretrained_weights(model, args.pretrained_weights.strip(), device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    checkpoint_path = save_dir / f"best_{tag}.pth"
    log_path = log_dir / f"{tag}_train_log.csv"

    print(f"开始训练 RAF-ML 多标签模型：{args.model}")
    print(f"实验标签：{tag}")
    print(f"训练方式：{training_mode}")
    if args.pretrained_weights.strip():
        print(f"预训练权重：{args.pretrained_weights}")
    print(f"当前设备：{device}")
    print(f"输出维度：{RAFML_NUM_LABELS}")
    print(f"sigmoid 阈值：{threshold}")
    print(f"可训练参数量：{count_parameters(model)}")
    print("损失函数：BCEWithLogitsLoss；预测时使用 sigmoid。")

    best_macro_f1 = -1.0
    best_epoch = 0
    history: list[dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_micro_f1, train_macro_f1 = run_one_epoch(
            model, train_loader, criterion, device, threshold, optimizer
        )
        test_loss, test_micro_f1, test_macro_f1 = run_one_epoch(
            model, test_loader, criterion, device, threshold
        )
        scheduler.step(test_loss)
        epoch_time = time.time() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_micro_f1": train_micro_f1,
                "train_macro_f1": train_macro_f1,
                "val_loss": test_loss,
                "val_micro_f1": test_micro_f1,
                "val_macro_f1": test_macro_f1,
                "lr": current_lr,
                "epoch_time": epoch_time,
            }
        )

        print(
            f"第 {epoch}/{args.epochs} 轮："
            f"train loss={train_loss:.4f}, train micro F1={train_micro_f1:.4f}, train macro F1={train_macro_f1:.4f}, "
            f"test loss={test_loss:.4f}, test micro F1={test_micro_f1:.4f}, test macro F1={test_macro_f1:.4f}, "
            f"lr={current_lr:.6f}, 用时={epoch_time:.1f}s"
        )

        if test_macro_f1 > best_macro_f1:
            best_macro_f1 = test_macro_f1
            best_epoch = epoch
            torch.save(
                {
                    "model_name": args.model,
                    "experiment_name": tag,
                    "state_dict": model.state_dict(),
                    "labels_en": RAFML_LABELS_EN,
                    "labels_cn": RAFML_LABELS_CN,
                    "num_labels": RAFML_NUM_LABELS,
                    "threshold": threshold,
                    "epoch": epoch,
                    "best_test_macro_f1": best_macro_f1,
                    "training_mode": training_mode,
                    "pretrained_weights": args.pretrained_weights,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"当前 test macro F1 最高，RAF-ML 多标签模型已保存：{checkpoint_path}")

    save_train_log(history, log_path)
    plot_rafml_training_curves(history, log_dir, tag)

    print("开始加载最佳 RAF-ML 多标签模型并计算最终指标...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = evaluate_metrics(model, test_loader, device, threshold)
    total_time = time.time() - start_time

    append_experiment_result(
        output_dir / "experiment_results.csv",
        {
            "model": tag,
            "use_aug": args.use_aug,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "best_epoch": best_epoch,
            "best_val_acc": f"{best_macro_f1:.6f}",
            "test_acc": f"{metrics['micro_f1']:.6f}",
            "macro_precision": f"{metrics['macro_precision']:.6f}",
            "macro_recall": f"{metrics['macro_recall']:.6f}",
            "macro_f1": f"{metrics['macro_f1']:.6f}",
            "parameters": count_parameters(model),
            "total_time_sec": f"{total_time:.2f}",
        },
    )

    print(f"训练完成，最佳 test macro F1：{best_macro_f1:.4f}，最佳轮次：{best_epoch}")
    print(f"最终 Micro Precision：{metrics['micro_precision']:.4f}")
    print(f"最终 Micro Recall：{metrics['micro_recall']:.4f}")
    print(f"最终 Micro F1：{metrics['micro_f1']:.4f}")
    print(f"最终 Macro Precision：{metrics['macro_precision']:.4f}")
    print(f"最终 Macro Recall：{metrics['macro_recall']:.4f}")
    print(f"最终 Macro F1：{metrics['macro_f1']:.4f}")
    print(f"最佳模型权重已保存到：{checkpoint_path}")
    print(f"训练日志已保存到：{log_path}")
    print(f"训练曲线已保存到：{log_dir}")
    print(f"实验摘要已追加保存到：{output_dir / 'experiment_results.csv'}")


if __name__ == "__main__":
    main()
