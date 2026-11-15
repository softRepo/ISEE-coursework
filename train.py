"""FER2013 模型训练脚本。"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn

from utils.common import CLASS_NAMES, build_model, count_parameters, ensure_dir, get_device, set_seed
from utils.dataset import get_fer2013_dataloaders
from utils.metrics import classification_metrics, confusion_matrix_np
from utils.visualize import plot_training_curves


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练 FER2013 人脸表情识别模型")
    parser.add_argument("--model", type=str, default="simple_cnn", choices=["simple_cnn", "improved_cnn", "mini_resnet"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=str, default="data/fer2013")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_aug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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

    for images, labels in loader:
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
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
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
    """保存每轮训练日志为 CSV。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time"])
        writer.writeheader()
        writer.writerows(history)


def append_experiment_result(result_path: Path, row: dict[str, object]) -> None:
    """追加保存实验摘要结果。"""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not result_path.exists()
    with result_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if need_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    save_dir = ensure_dir(args.save_dir)
    log_dir = ensure_dir(args.log_dir)
    output_dir = ensure_dir(args.output_dir)

    print("开始加载 FER2013 数据集...")
    train_loader, val_loader, test_loader = get_fer2013_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_aug=args.use_aug,
        seed=args.seed,
    )
    print(f"训练集样本数：{len(train_loader.dataset)}")
    print(f"验证集样本数：{len(val_loader.dataset)}")
    print(f"测试集样本数：{len(test_loader.dataset)}")

    device = get_device()
    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"开始训练模型：{args.model}")
    print(f"当前设备：{device}")
    print(f"可训练参数量：{count_parameters(model)}")

    best_val_acc = -1.0
    best_epoch = 0
    history: list[dict[str, float]] = []
    checkpoint_path = save_dir / f"best_{args.model}.pth"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time": epoch_time,
            }
        )

        print(
            f"第 {epoch}/{args.epochs} 轮："
            f"train loss={train_loss:.4f}, train acc={train_acc:.4f}, "
            f"val loss={val_loss:.4f}, val acc={val_acc:.4f}, 用时={epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"当前验证集准确率最高，模型已保存：{checkpoint_path}")

    save_train_log(history, log_dir / f"{args.model}_train_log.csv")
    plot_training_curves(history, log_dir, args.model)

    print("开始加载最佳模型并在测试集上评估...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_acc, test_precision, test_recall, test_f1 = evaluate_loader(model, test_loader, device)
    total_time = time.time() - start_time

    result_row = {
        "model": args.model,
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
    print(f"测试集准确率：{test_acc:.4f}，宏平均 F1：{test_f1:.4f}")
    print(f"训练日志与曲线已保存到：{log_dir}")
    print(f"实验摘要已保存到：{output_dir / 'experiment_results.csv'}")


if __name__ == "__main__":
    main()
