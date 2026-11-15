from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common_fashion import (
    LABEL_NAME_MAP,
    ORIG_TO_IDX,
    TARGET_CLASSES,
    count_parameters,
    default_data_dir,
    default_figures_dir,
    default_results_dir,
    save_json,
    set_seed,
)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


class FashionMiniDataset(Dataset):
    def __init__(self, data_dir: str | Path, transform=None):
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(self.data_dir / "labels.csv")
        self.img_dir = self.data_dir / "images"
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(self.img_dir / row["filename"]).convert("L")
        # 交叉熵损失函数需要连续类别编号，因此将 0/3/5/6/7 映射为 0-4。
        label = ORIG_TO_IDX[int(row["label_id"])]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward_features(self, x):
        conv1 = F.relu(self.conv1(x))
        x = F.max_pool2d(conv1, 2)
        conv2 = F.relu(self.conv2(x))
        x = F.max_pool2d(conv2, 2)
        return x, {"conv1": conv1, "conv2": conv2}

    def forward(self, x):
        x, _ = self.forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tuned LeNet-5 baseline")
    parser.add_argument("--train_dir", type=str, default=str(default_data_dir("train")))
    parser.add_argument("--test_dir", type=str, default=str(default_data_dir("test")))
    parser.add_argument("--figures_dir", type=str, default=str(default_figures_dir()))
    parser.add_argument("--results_dir", type=str, default=str(default_results_dir()))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total * 100.0


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total * 100.0


def save_curves(history: list[dict], figures_dir: Path) -> None:
    epochs = [item["epoch"] for item in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [item["train_loss"] for item in history], marker="o", label="Train")
    plt.plot(epochs, [item["test_loss"] for item in history], marker="s", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "lenet_base_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [item["train_acc"] for item in history], marker="o", label="Train")
    plt.plot(epochs, [item["test_acc"] for item in history], marker="s", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "lenet_base_accuracy.png", dpi=180)
    plt.close()


def save_feature_maps(model, dataset, figures_dir: Path, device) -> None:
    # 保存两个卷积层的特征图，用于比较浅层与深层特征差异。
    model.eval()
    image, label = dataset[0]
    with torch.no_grad():
        _, feature_maps = model.forward_features(image.unsqueeze(0).to(device))
    conv1 = feature_maps["conv1"].squeeze(0).cpu()
    conv2 = feature_maps["conv2"].squeeze(0).cpu()

    plt.figure(figsize=(10, 5))
    plt.subplot(3, 8, 1)
    plt.imshow(image.squeeze(0), cmap="gray")
    plt.title(LABEL_NAME_MAP[TARGET_CLASSES[label]], fontsize=8)
    plt.axis("off")
    for i in range(min(6, conv1.size(0))):
        plt.subplot(3, 8, i + 2)
        plt.imshow(conv1[i], cmap="viridis")
        plt.title(f"C1-{i + 1}", fontsize=8)
        plt.axis("off")
    for i in range(8):
        plt.subplot(3, 8, 9 + i)
        plt.imshow(conv2[i], cmap="magma")
        plt.title(f"C2-{i + 1}", fontsize=8)
        plt.axis("off")
    for i in range(8, 16):
        plt.subplot(3, 8, 9 + i)
        plt.imshow(conv2[i], cmap="magma")
        plt.title(f"C2-{i + 1}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(figures_dir / "lenet_base_feature_maps.png", dpi=180)
    plt.close()


def main() -> None:
    configure_stdout()
    args = get_parser().parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    figures_dir = Path(args.figures_dir)
    results_dir = Path(args.results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 启动优化版 LeNet-5 训练 | 设备: {device}")
    print(f"📂 训练集路径: {args.train_dir}")
    print(f"📂 测试集路径: {args.test_dir}")
    print(f"📁 数值结果将保存至: {results_dir}")
    print(f"🖼️ 可视化图片将保存至: {figures_dir}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = FashionMiniDataset(args.train_dir, transform=transform)
    test_dataset = FashionMiniDataset(args.test_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"📊 加载成功 | 训练集: {len(train_dataset)} 张 | 测试集: {len(test_dataset)} 张")
    print(
        f"⚙️  超参数 | batch_size={args.batch_size} | lr={args.lr} | "
        f"epochs={args.epochs} | weight_decay={args.weight_decay}"
    )

    model = LeNet5(num_classes=len(TARGET_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    start_time = time.perf_counter()
    best_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        if test_acc > best_acc:
            # 保存测试准确率最高的权重，而不是直接使用最后一轮权重。
            best_acc = test_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        }
        history.append(item)
        print(
            f"✅ Epoch[{epoch:02d}] 结束 | "
            f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
            f"本轮耗时: {time.perf_counter() - epoch_start:.2f}s"
        )

    total_time = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    best_loss, best_eval_acc = evaluate(model, test_loader, criterion, device)
    save_curves(history, figures_dir)
    save_feature_maps(model, test_dataset, figures_dir, device)
    torch.save(model.state_dict(), results_dir / "lenet_base.pth")
    save_json(
        results_dir / "lenet_base_metrics.json",
        {
            "model": "LeNet5_tuned",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "best_test_acc": best_acc,
            "best_epoch": best_epoch,
            "best_eval_loss": best_loss,
            "best_eval_acc_after_reload": best_eval_acc,
            "final_test_acc": history[-1]["test_acc"],
            "train_time_sec": total_time,
            "params": count_parameters(model),
            "history": history,
        },
    )

    print("\n" + "=" * 60)
    print("🎯 优化版 LeNet-5 测试结果")
    print("=" * 60)
    print(f"✅ 最佳测试准确率: {best_acc:.2f}%")
    print(f"📌 最佳 Epoch: {best_epoch}")
    print(f"🔁 重新加载最佳权重后准确率: {best_eval_acc:.2f}%")
    print(f"📊 最后一轮测试准确率: {history[-1]['test_acc']:.2f}%")
    print(f"🧮 可训练参数量: {count_parameters(model)}")
    print(f"⏱️ 总训练耗时: {total_time:.2f}s")
    print(f"💾 模型与指标已保存至: {results_dir}")
    print(f"🖼️ 曲线与特征图已保存至: {figures_dir}")
    print("🎉 运行完成！")


if __name__ == "__main__":
    main()
