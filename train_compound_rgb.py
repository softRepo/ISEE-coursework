"""RAF-DB Compound 复合表情 RGB 模型训练脚本。"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn

from models.rgb_mini_resnet import RGBMiniResNet
from utils.common import count_parameters, ensure_dir, get_device, set_seed
from utils.compound_dataset import (
    COMPOUND_LABELS_CN,
    COMPOUND_LABELS_EN,
    COMPOUND_NUM_CLASSES,
    get_compound_dataloaders,
)
from utils.metrics import classification_metrics, confusion_matrix_np
from utils.visualize import plot_training_curves

DEFAULT_TAG = "compound_rgb_weighted"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="训练 RAF-DB Compound 11 类复合表情 RGB 模型")
    parser.add_argument("--data_dir", type=str, default="data/compound")
    parser.add_argument(
        "--model",
        type=str,
        default="compound_rgb_mini_resnet",
        choices=["compound_rgb_mini_resnet", "rgb_mini_resnet"],
    )
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
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="",
        help="可选 RGB 预训练权重路径，只加载 backbone/特征层，跳过 11 类分类层",
    )
    parser.add_argument("--dropout", type=float, default=0.4, help="复合表情模型分类头 dropout，默认 0.4 用于缓解过拟合")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="标签平滑系数，默认 0.1 用于降低过拟合")
    parser.add_argument("--freeze_backbone", action="store_true", help="使用预训练权重时冻结 stem/layer1/layer2，只训练后面 block 和分类层")
    parser.add_argument("--early_stopping_patience", type=int, default=8, help="测试集准确率连续多少轮不提升后提前停止，0 表示关闭")
    return parser.parse_args()


def build_compound_model(model_name: str) -> nn.Module:
    """创建 11 类复合表情 RGB MiniResNet。"""
    if model_name in {"compound_rgb_mini_resnet", "rgb_mini_resnet"}:
        return RGBMiniResNet(num_classes=COMPOUND_NUM_CLASSES)
    raise ValueError("未知模型名称，目前仅支持 compound_rgb_mini_resnet 或 rgb_mini_resnet。")


def set_classifier_dropout(model: nn.Module, dropout: float) -> None:
    """只调整复合表情实验分类头的 dropout，不改变原始模型定义。"""
    if dropout < 0.4:
        print("提示：复合表情实验 dropout 不低于 0.4，已自动调整为 0.4。")
    dropout = max(0.4, min(float(dropout), 0.9))
    if hasattr(model, "classifier"):
        for module in model.classifier:
            if isinstance(module, nn.Dropout):
                module.p = dropout
                print(f"复合表情分类头 Dropout 已设置为：{dropout:.2f}")
                return
    print("提示：当前模型未找到分类头 Dropout，已跳过 dropout 调整。")


def compute_class_weights(train_loader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
    """根据训练集每类样本数自动计算类别权重。"""
    counts = torch.zeros(COMPOUND_NUM_CLASSES, dtype=torch.float32)
    samples = getattr(train_loader.dataset, "samples", [])
    for sample in samples:
        counts[int(sample.label)] += 1

    if int(counts.sum().item()) == 0:
        raise RuntimeError("训练集样本数量为 0，无法计算类别权重。")

    weights = torch.zeros_like(counts)
    nonzero_mask = counts > 0
    weights[nonzero_mask] = counts.sum() / (COMPOUND_NUM_CLASSES * counts[nonzero_mask])
    # 将非零类别权重均值归一到 1，便于 loss 数值保持稳定。
    weights[nonzero_mask] = weights[nonzero_mask] / weights[nonzero_mask].mean()

    print("复合表情类别权重：")
    for index, (count, weight) in enumerate(zip(counts.tolist(), weights.tolist())):
        print(f"  {index} {COMPOUND_LABELS_CN[index]}：样本数 {int(count)}，权重 {weight:.4f}")
    return weights.to(device)


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


def freeze_early_backbone_layers(model: nn.Module) -> None:
    """冻结前面的卷积特征层，只训练后面的 block 和分类层。"""
    frozen_prefixes = ("stem.", "layer1.", "layer2.")
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if name.startswith(frozen_prefixes):
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = True
            trainable_count += param.numel()
    print("已启用 freeze_backbone：冻结 stem/layer1/layer2，只训练 layer3/layer4/classifier。")
    print(f"  冻结参数量：{frozen_count}")
    print(f"  可训练参数量：{trainable_count}")


def run_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """执行一个训练或测试轮次。"""
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


def evaluate_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """计算复合表情测试集上的总体指标。"""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    with torch.no_grad():
        for images, labels, _ in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    matrix = confusion_matrix_np(all_labels, all_preds, COMPOUND_NUM_CLASSES)
    metrics = classification_metrics(matrix)
    return (
        metrics["accuracy"],
        metrics["macro_precision"],
        metrics["macro_recall"],
        metrics["macro_f1"],
    )


def save_train_log(history: list[dict[str, float]], log_path: Path) -> None:
    """保存复合表情训练日志为 CSV 文件。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time"],
        )
        writer.writeheader()
        writer.writerows(history)


def append_experiment_result(result_path: Path, row: dict[str, object]) -> None:
    """将复合表情实验摘要追加到统一实验结果表。"""
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
    """训练 RAF-DB Compound 复合表情 RGB 模型。"""
    args = parse_args()
    set_seed(args.seed)

    save_dir = ensure_dir(args.save_dir)
    log_dir = ensure_dir(args.log_dir)
    output_dir = ensure_dir(args.output_dir)
    tag = args.tag.strip() or DEFAULT_TAG

    print("开始加载 RAF-DB Compound 复合表情数据集...")
    train_loader, test_loader = get_compound_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_aug=args.use_aug,
        seed=args.seed,
    )

    device = get_device()
    model = build_compound_model(args.model).to(device)
    set_classifier_dropout(model, args.dropout)
    training_mode = "预训练微调" if args.pretrained_weights.strip() else "从头训练"
    if args.pretrained_weights.strip():
        load_backbone_pretrained_weights(model, args.pretrained_weights.strip(), device)
        if args.freeze_backbone:
            freeze_early_backbone_layers(model)
    elif args.freeze_backbone:
        print("提示：未传入 --pretrained_weights，--freeze_backbone 已忽略，当前仍为从头训练。")

    class_weights = compute_class_weights(train_loader, device)
    label_smoothing = max(0.0, min(float(args.label_smoothing), 0.5))
    if label_smoothing != args.label_smoothing:
        print(f"提示：label_smoothing 已从 {args.label_smoothing} 调整为 {label_smoothing}。")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("当前没有可训练参数，请检查 freeze_backbone 设置。")
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    checkpoint_path = save_dir / f"best_{tag}.pth"
    log_path = log_dir / f"{tag}_train_log.csv"

    print(f"开始训练 RAF-DB Compound 复合表情模型：{args.model}")
    print(f"实验标签：{tag}")
    print(f"训练方式：{training_mode}")
    if args.pretrained_weights.strip():
        print(f"预训练权重：{args.pretrained_weights}")
    print(f"损失函数：CrossEntropyLoss，已启用 class weights 和 label_smoothing={label_smoothing}")
    print(f"正则化：weight_decay={args.weight_decay}，dropout 参数={args.dropout}，实际 dropout 不低于 0.4")
    print(f"Early Stopping patience：{args.early_stopping_patience}")
    print(f"当前设备：{device}")
    print(f"类别数量：{COMPOUND_NUM_CLASSES}")
    print(f"可训练参数量：{count_parameters(model)}")
    print("说明：本脚本使用官方 train/test 划分；test 指标用于扩展实验记录。")

    best_test_acc = -1.0
    best_epoch = 0
    no_improve_epochs = 0
    history: list[dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_acc = run_one_epoch(model, test_loader, criterion, device)
        scheduler.step(test_loss)
        epoch_time = time.time() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": test_loss,
                "val_acc": test_acc,
                "lr": current_lr,
                "epoch_time": epoch_time,
            }
        )

        print(
            f"第 {epoch}/{args.epochs} 轮："
            f"train loss={train_loss:.4f}, train acc={train_acc:.4f}, "
            f"test loss={test_loss:.4f}, test acc={test_acc:.4f}, "
            f"lr={current_lr:.6f}, 用时={epoch_time:.1f}s"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(
                {
                    "model_name": args.model,
                    "experiment_name": tag,
                    "state_dict": model.state_dict(),
                    "class_names_en": COMPOUND_LABELS_EN,
                    "class_names_cn": COMPOUND_LABELS_CN,
                    "num_classes": COMPOUND_NUM_CLASSES,
                    "epoch": epoch,
                    "best_test_acc": best_test_acc,
                    "training_mode": training_mode,
                    "pretrained_weights": args.pretrained_weights,
                    "class_weights": class_weights.detach().cpu().tolist(),
                    "label_smoothing": label_smoothing,
                    "freeze_backbone": args.freeze_backbone,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"当前测试集准确率最高，复合表情模型已保存：{checkpoint_path}")
        else:
            no_improve_epochs += 1
            if args.early_stopping_patience > 0:
                print(f"测试集准确率已连续 {no_improve_epochs} 轮未提升。")
                if no_improve_epochs >= args.early_stopping_patience:
                    print(
                        f"触发 Early Stopping：连续 {args.early_stopping_patience} 轮未提升，"
                        f"提前停止训练。最佳轮次：{best_epoch}"
                    )
                    break

    save_train_log(history, log_path)
    plot_training_curves(history, log_dir, tag)

    print("开始加载最佳复合表情模型并计算最终指标...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_acc, test_precision, test_recall, test_f1 = evaluate_metrics(model, test_loader, device)
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
            "best_val_acc": f"{best_test_acc:.6f}",
            "test_acc": f"{test_acc:.6f}",
            "macro_precision": f"{test_precision:.6f}",
            "macro_recall": f"{test_recall:.6f}",
            "macro_f1": f"{test_f1:.6f}",
            "parameters": count_parameters(model),
            "total_time_sec": f"{total_time:.2f}",
        },
    )

    print(f"训练完成，最佳测试集准确率：{best_test_acc:.4f}，最佳轮次：{best_epoch}")
    print(f"最终 Accuracy：{test_acc:.4f}")
    print(f"最终 Macro Precision：{test_precision:.4f}")
    print(f"最终 Macro Recall：{test_recall:.4f}")
    print(f"最终 Macro F1-score：{test_f1:.4f}")
    print(f"最佳模型权重已保存到：{checkpoint_path}")
    print(f"训练日志已保存到：{log_path}")
    print(f"训练曲线已保存到：{log_dir}")
    print(f"实验摘要已追加保存到：{output_dir / 'experiment_results.csv'}")


if __name__ == "__main__":
    main()
