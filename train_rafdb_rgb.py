"""RAF-DB RGB 模型训练脚本。"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn

from models.rgb_mini_resnet import RGBMiniResNet
from utils.common import CLASS_NAMES, count_parameters, ensure_dir, get_device, set_seed
from utils.metrics import classification_metrics, confusion_matrix_np
from utils.rafdb_dataset import get_rafdb_rgb_dataloaders
from utils.visualize import plot_training_curves

EXPERIMENT_NAME = "rafdb_rgb_mini_resnet"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="从头训练 RAF-DB RGB 表情识别模型")
    parser.add_argument("--data_dir", type=str, default="data/RAF-DB/DATASET")
    parser.add_argument("--model", type=str, default="rgb_mini_resnet", choices=["rgb_mini_resnet"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--use_aug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.15, help="从训练集中划分验证集的比例")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    return parser.parse_args()


def build_rgb_model(model_name: str) -> nn.Module:
    """根据模型名称创建 RAF-DB RGB 模型。"""
    if model_name == "rgb_mini_resnet":
        return RGBMiniResNet(num_classes=len(CLASS_NAMES))
    raise ValueError("未知模型名称，目前仅支持 rgb_mini_resnet。")


def run_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """执行一个训练或验证轮次。"""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0

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
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """在给定数据集上计算总体指标。"""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    with torch.no_grad():
        for images, labels, _ in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    matrix = confusion_matrix_np(all_labels, all_preds, len(CLASS_NAMES))
    metrics = classification_metrics(matrix)
    return (
        metrics["accuracy"],
        metrics["macro_precision"],
        metrics["macro_recall"],
        metrics["macro_f1"],
    )


def save_train_log(history: list[dict[str, float]], log_path: Path) -> None:
    """保存训练日志为 CSV 文件。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time"],
        )
        writer.writeheader()
        writer.writerows(history)


def append_experiment_result(result_path: Path, row: dict[str, object]) -> None:
    """将 RAF-DB RGB 实验摘要追加到统一实验结果表。"""
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
    args = parse_args()
    set_seed(args.seed)

    save_dir = ensure_dir(args.save_dir)
    log_dir = ensure_dir(args.log_dir)
    output_dir = ensure_dir(args.output_dir)

    print("开始加载 RAF-DB RGB 数据集...")
    train_loader, val_loader, test_loader = get_rafdb_rgb_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_aug=args.use_aug,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )

    device = get_device()
    model = build_rgb_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    print(f"开始训练 RAF-DB RGB 模型：{args.model}")
    print(f"当前设备：{device}")
    print(f"可训练参数量：{count_parameters(model)}")
    print("说明：RAF-DB test 只用于训练完成后的最终评估，不用于选择最佳模型。")

    best_val_acc = -1.0
    best_epoch = 0
    history: list[dict[str, float]] = []
    checkpoint_path = save_dir / "best_rafdb_rgb_mini_resnet.pth"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
                "epoch_time": epoch_time,
            }
        )

        print(
            f"第 {epoch}/{args.epochs} 轮："
            f"train loss={train_loss:.4f}, train acc={train_acc:.4f}, "
            f"val loss={val_loss:.4f}, val acc={val_acc:.4f}, "
            f"lr={current_lr:.6f}, 用时={epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_name": args.model,
                    "experiment_name": EXPERIMENT_NAME,
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"当前验证集准确率最高，RAF-DB RGB 模型已保存：{checkpoint_path}")

    save_train_log(history, log_dir / f"{EXPERIMENT_NAME}_train_log.csv")
    plot_training_curves(history, log_dir, EXPERIMENT_NAME)

    print("开始加载最佳 RAF-DB RGB 模型并在 test 集上最终评估...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_acc, test_precision, test_recall, test_f1 = evaluate_loader(model, test_loader, device)
    total_time = time.time() - start_time

    result_row = {
        "model": EXPERIMENT_NAME,
        "use_aug": args.use_aug,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_epoch": best_epoch,
        "best_val_acc": f"{best_val_acc:.6f}",
        "test_acc": f"{test_acc:.6f}",
        "macro_precision": f"{test_precision:.6f}",
        "macro_recall": f"{test_recall:.6f}",
        "macro_f1": f"{test_f1:.6f}",
        "parameters": count_parameters(model),
        "total_time_sec": f"{total_time:.2f}",
    }
    append_experiment_result(output_dir / "experiment_results.csv", result_row)

    print(f"训练完成，最佳验证集准确率：{best_val_acc:.4f}，最佳轮次：{best_epoch}")
    print(f"RAF-DB test 准确率：{test_acc:.4f}，宏平均 F1：{test_f1:.4f}")
    print(f"最佳模型权重已保存到：{checkpoint_path}")
    print(f"训练日志与曲线已保存到：{log_dir}")
    print(f"实验摘要已追加保存到：{output_dir / 'experiment_results.csv'}")


if __name__ == "__main__":
    main()
