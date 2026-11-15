from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB

from common_fashion import (
    LABEL_NAME_MAP,
    TARGET_CLASSES,
    default_data_dir,
    default_figures_dir,
    default_results_dir,
    load_flat_data,
    save_json,
    set_seed,
)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bernoulli Naive Bayes for Fashion-MNIST subset")
    parser.add_argument("--train_dir", type=str, default=str(default_data_dir("train")))
    parser.add_argument("--test_dir", type=str, default=str(default_data_dir("test")))
    parser.add_argument("--results_dir", type=str, default=str(default_results_dir()))
    parser.add_argument("--figures_dir", type=str, default=str(default_figures_dir()))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.50, 0.65])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--random_state", type=int, default=42)
    return parser


def class_stats(y_true, y_pred) -> dict[int, dict[str, float]]:
    stats = {}
    for label in TARGET_CLASSES:
        mask = y_true == label
        total = int(mask.sum())
        correct = int((y_pred[mask] == y_true[mask]).sum()) if total else 0
        acc = 100.0 * correct / total if total else 0.0
        stats[label] = {"correct": correct, "total": total, "accuracy": acc}
    return stats


def evaluate(name: str, model, x_train, y_train, x_test, y_test) -> dict:
    # 与 GaussianNB 保持相同的评价流程，保证对比结果公平。
    start = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    pred = model.predict(x_test)
    infer_time = time.perf_counter() - start

    return {
        "method": name,
        "accuracy": accuracy_score(y_test, pred) * 100.0,
        "train_time_sec": train_time,
        "test_time_sec": infer_time,
        "class_stats": class_stats(y_test, pred),
    }


def save_accuracy_plot(rows: list[dict], figures_dir: str | Path) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    methods = [row["method"] for row in rows]
    values = [float(row["accuracy"].rstrip("%")) for row in rows]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(range(len(values)), values, color="#59A14F")
    plt.xticks(range(len(values)), methods, rotation=25, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, max(values) + 8)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "bayes_bernoulli_accuracy.png", dpi=180)
    plt.close()


def print_class_stats(stats: dict[int, dict[str, float]]) -> None:
    print("\n📊 最佳配置各类别正确率：")
    for label in TARGET_CLASSES:
        item = stats[label]
        print(
            f"   类别{label:2d} ({LABEL_NAME_MAP[label]:10s}): "
            f"{item['correct']}/{item['total']} | 准确率={item['accuracy']:.2f}%"
        )


def main() -> None:
    configure_stdout()
    args = get_parser().parse_args()
    set_seed(args.random_state)

    print("🚀 启动 伯努利朴素贝叶斯 对比实验")
    print(f"📂 训练集路径: {args.train_dir}")
    print(f"📂 测试集路径: {args.test_dir}")

    x_train, y_train = load_flat_data(args.train_dir, scale_01=True)
    x_test, y_test = load_flat_data(args.test_dir, scale_01=True)
    print(f"📊 训练集: {len(x_train)} 张 | 测试集: {len(x_test)} 张")

    results = [
        evaluate("GaussianNB_raw", GaussianNB(var_smoothing=1e-9), x_train, y_train, x_test, y_test)
    ]

    # 二值化参数将灰度像素转换为前景/背景二值特征。
    for threshold in args.thresholds:
        print(f"🔧 正在运行 BernoulliNB 配置: binarize={threshold:.2f}")
        model = BernoulliNB(alpha=args.alpha, binarize=threshold)
        results.append(evaluate(f"BernoulliNB_thr_{threshold:.2f}", model, x_train, y_train, x_test, y_test))

    rows = [
        {
            "method": r["method"],
            "accuracy": f"{r['accuracy']:.2f}%",
            "train_s": f"{r['train_time_sec']:.3f}",
            "test_s": f"{r['test_time_sec']:.3f}",
        }
        for r in results
    ]

    print("\n" + "=" * 60)
    print("🎯 伯努利朴素贝叶斯 测试结果")
    print("=" * 60)
    print("📌 对比结果：")
    for row in rows:
        print(
            f"   {row['method']:22s} | 准确率={row['accuracy']:>7s} | "
            f"训练={row['train_s']}s | 测试={row['test_s']}s"
        )

    best = max(results, key=lambda item: item["accuracy"])
    print(f"\n🏆 最佳配置: {best['method']} | 测试准确率: {best['accuracy']:.2f}%")
    print_class_stats(best["class_stats"])

    results_dir = Path(args.results_dir)
    pd.DataFrame(rows).to_csv(results_dir / "bayes_bernoulli_results.csv", index=False, encoding="utf-8-sig")
    save_json(results_dir / "bayes_bernoulli_results.json", {"results": results})
    save_accuracy_plot(rows, args.figures_dir)

    print(f"\n💾 数值结果已保存至: {results_dir}")
    print(f"🖼️ 可视化图片已保存至: {args.figures_dir}")
    print("🎉 运行完成！")


if __name__ == "__main__":
    main()
