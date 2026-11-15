from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from common_fashion import (
    count_parameters,
    default_data_dir,
    default_figures_dir,
    default_results_dir,
    save_json,
    set_seed,
)
from lenet_base import FashionMiniDataset, evaluate, train_one_epoch


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # 主分支学习特征修正，shortcut 分支保留输入信息，便于梯度传播。
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResidualFashionCNN(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(32, 32, dropout=0.05)
        self.layer2 = ResidualBlock(32, 64, stride=2, dropout=0.05)
        self.layer3 = ResidualBlock(64, 128, stride=2, dropout=0.10)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(128, num_classes))

    def forward_features(self, x):
        stem = self.stem(x)
        x = self.layer1(stem)
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        return layer3, {"stem": stem, "layer2": layer2, "layer3": layer3}

    def forward(self, x):
        x, _ = self.forward_features(x)
        x = self.pool(x)
        return self.classifier(x)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Improved residual CNN with ablation study")
    parser.add_argument("--train_dir", type=str, default=str(default_data_dir("train")))
    parser.add_argument("--test_dir", type=str, default=str(default_data_dir("test")))
    parser.add_argument("--figures_dir", type=str, default=str(default_figures_dir()))
    parser.add_argument("--results_dir", type=str, default=str(default_results_dir()))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def make_transforms(use_augmentation: bool):
    base = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    if not use_augmentation:
        return transforms.Compose(base)
    # 轻量数据增强用于检验几何扰动是否能够提升模型泛化能力。
    return transforms.Compose(
        [
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            *base,
        ]
    )


def run_experiment(name: str, args, device, use_augmentation: bool) -> tuple[nn.Module, list[dict], dict]:
    train_dataset = FashionMiniDataset(args.train_dir, transform=make_transforms(use_augmentation))
    test_dataset = FashionMiniDataset(args.test_dir, transform=make_transforms(False))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ResidualFashionCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\n" + "=" * 60)
    print(f"🎯 开始训练配置: {name}")
    print("=" * 60)
    print(f"📌 数据增强: {'开启' if use_augmentation else '关闭'}")
    print(f"📊 训练样本: {len(train_dataset)} | 测试样本: {len(test_dataset)}")
    print(f"🔢 参数量: {count_parameters(model):,}")

    history = []
    best_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        if test_acc > best_acc:
            # 每个消融配置都单独保存测试准确率最高的权重。
            best_acc = test_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )
        print(
            f"✅ {name} Epoch[{epoch:02d}/{args.epochs:02d}] 结束 | "
            f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | "
            f"本轮耗时: {time.perf_counter() - epoch_start:.1f}s"
        )

    model.load_state_dict(best_state)
    best_loss, best_eval_acc = evaluate(model, test_loader, criterion, device)
    metrics = {
        "model": name,
        "use_augmentation": use_augmentation,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "best_eval_loss": best_loss,
        "best_eval_acc_after_reload": best_eval_acc,
        "final_test_acc": history[-1]["test_acc"],
        "train_time_sec": time.perf_counter() - start_time,
        "params": count_parameters(model),
        "history": history,
    }
    print(
        f"🏆 {name} 最佳测试准确率: {best_acc:.2f}% | "
        f"最佳 Epoch: {best_epoch} | 训练耗时: {metrics['train_time_sec']:.1f}s"
    )
    return model, history, metrics


def save_ablation_curves(histories: dict[str, list[dict]], figures_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    for name, history in histories.items():
        plt.plot([h["epoch"] for h in history], [h["test_acc"] for h in history], marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "lenet_plus_ablation_accuracy.png", dpi=180)
    plt.close()

    best_name = max(histories, key=lambda n: max(item["test_acc"] for item in histories[n]))
    history = histories[best_name]
    epochs = [item["epoch"] for item in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [item["train_loss"] for item in history], marker="o", label="Train")
    plt.plot(epochs, [item["test_loss"] for item in history], marker="s", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "lenet_plus_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [item["train_acc"] for item in history], marker="o", label="Train")
    plt.plot(epochs, [item["test_acc"] for item in history], marker="s", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "lenet_plus_accuracy.png", dpi=180)
    plt.close()


def save_feature_maps(model, test_dir: str, figures_dir: Path, device) -> None:
    # 可视化不同深度的特征图，观察低级纹理到高级响应的变化。
    dataset = FashionMiniDataset(test_dir, transform=make_transforms(False))
    image, _ = dataset[0]
    model.eval()
    with torch.no_grad():
        _, feature_maps = model.forward_features(image.unsqueeze(0).to(device))

    stages = [("Stem", feature_maps["stem"]), ("Layer2", feature_maps["layer2"]), ("Layer3", feature_maps["layer3"])]
    plt.figure(figsize=(10, 6))
    for row, (stage_name, fmap) in enumerate(stages):
        fmap = fmap.squeeze(0).cpu()
        for col in range(8):
            plt.subplot(3, 8, row * 8 + col + 1)
            plt.imshow(fmap[col], cmap="magma")
            plt.title(f"{stage_name}-{col + 1}", fontsize=7)
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(figures_dir / "lenet_plus_feature_maps.png", dpi=180)
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
    experiments = [
        ("ResidualCNN_no_aug", False),
        ("ResidualCNN_aug", True),
    ]

    print(f"[{time.strftime('%H:%M:%S')}] 🚀 启动 ResidualCNN 改进与消融实验 | 设备: {device}")
    print(f"📂 训练集路径: {Path(args.train_dir).resolve()}")
    print(f"📂 测试集路径: {Path(args.test_dir).resolve()}")
    print(f"📁 结果保存路径: {results_dir.resolve()}")
    print(f"🖼️ 图片保存路径: {figures_dir.resolve()}")
    print(
        f"⚙️ 超参数: epochs={args.epochs}, batch_size={args.batch_size}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, seed={args.seed}"
    )

    histories = {}
    all_metrics = []
    best_model = None
    best_acc = -1.0
    for name, use_aug in experiments:
        model, history, metrics = run_experiment(name, args, device, use_aug)
        histories[name] = history
        all_metrics.append(metrics)
        if metrics["best_test_acc"] > best_acc:
            best_acc = metrics["best_test_acc"]
            best_model = model

    save_ablation_curves(histories, figures_dir)
    save_feature_maps(best_model, args.test_dir, figures_dir, device)
    torch.save(best_model.state_dict(), results_dir / "lenet_plus_best.pth")
    save_json(results_dir / "lenet_plus_metrics.json", {"experiments": all_metrics})

    print("\n" + "=" * 60)
    print("🎯 ResidualCNN 消融实验测试结果")
    print("=" * 60)
    for metrics in all_metrics:
        print(
            f"{metrics['model']}: 最佳测试准确率 {metrics['best_test_acc']:.2f}% | "
            f"最终测试准确率 {metrics['final_test_acc']:.2f}% | "
            f"最佳 Epoch {metrics['best_epoch']} | 参数量 {metrics['params']:,}"
        )
    print(f"🏆 最优配置测试准确率: {best_acc:.2f}%")
    print(f"💾 模型文件: {results_dir / 'lenet_plus_best.pth'}")
    print(f"📄 指标文件: {results_dir / 'lenet_plus_metrics.json'}")
    print(f"📈 消融曲线: {figures_dir / 'lenet_plus_ablation_accuracy.png'}")
    print(f"📈 Loss 曲线: {figures_dir / 'lenet_plus_loss.png'}")
    print(f"📈 Accuracy 曲线: {figures_dir / 'lenet_plus_accuracy.png'}")
    print(f"🖼️ 特征图: {figures_dir / 'lenet_plus_feature_maps.png'}")
    print("🎉 运行完成！")


if __name__ == "__main__":
    main()
