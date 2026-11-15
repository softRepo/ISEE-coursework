from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    parser = argparse.ArgumentParser(description="Gaussian Naive Bayes with PCA for Fashion-MNIST subset")
    parser.add_argument("--train_dir", type=str, default=str(default_data_dir("train")))
    parser.add_argument("--test_dir", type=str, default=str(default_data_dir("test")))
    parser.add_argument("--results_dir", type=str, default=str(default_results_dir()))
    parser.add_argument("--figures_dir", type=str, default=str(default_figures_dir()))
    parser.add_argument("--components", type=int, nargs="+", default=[20, 50, 100, 150, 200])
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


def evaluate_model(name: str, model, x_train, y_train, x_test, y_test) -> dict:
    # 对所有配置采用同一套计时方式，便于公平比较训练和推理耗时。
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
    bars = plt.bar(range(len(values)), values, color="#4C78A8")
    plt.xticks(range(len(values)), methods, rotation=25, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, max(values) + 8)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "bayes_pca_accuracy.png", dpi=180)
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

    print("🚀 启动 PCA + 高斯朴素贝叶斯 对比实验")
    print(f"📂 训练集路径: {args.train_dir}")
    print(f"📂 测试集路径: {args.test_dir}")

    x_train, y_train = load_flat_data(args.train_dir, scale_01=True)
    x_test, y_test = load_flat_data(args.test_dir, scale_01=True)
    print(f"📊 训练集: {len(x_train)} 张 | 测试集: {len(x_test)} 张")

    results = [
        evaluate_model("GaussianNB_raw_784d", GaussianNB(var_smoothing=1e-9), x_train, y_train, x_test, y_test)
    ]

    # PCA 只在训练集上拟合，再将同一个投影变换应用到测试集，避免数据泄漏。
    for n_components in args.components:
        print(f"🔧 正在运行 PCA 降维配置: n_components={n_components}")
        pca_nb = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "pca",
                    PCA(n_components=n_components, whiten=True, random_state=args.random_state),
                ),
                ("gnb", GaussianNB(var_smoothing=1e-9)),
            ]
        )
        result = evaluate_model(f"GaussianNB_PCA_{n_components}d", pca_nb, x_train, y_train, x_test, y_test)
        pca = pca_nb.named_steps["pca"]
        result["explained_variance_ratio"] = float(pca.explained_variance_ratio_.sum())
        results.append(result)

    rows = [
        {
            "method": r["method"],
            "accuracy": f"{r['accuracy']:.2f}%",
            "train_s": f"{r['train_time_sec']:.3f}",
            "test_s": f"{r['test_time_sec']:.3f}",
            "variance": f"{r.get('explained_variance_ratio', 1.0):.4f}",
        }
        for r in results
    ]

    print("\n" + "=" * 60)
    print("🎯 PCA + 高斯朴素贝叶斯 测试结果")
    print("=" * 60)
    print("📌 对比结果：")
    for row in rows:
        print(
            f"   {row['method']:22s} | 准确率={row['accuracy']:>7s} | "
            f"训练={row['train_s']}s | 测试={row['test_s']}s | 方差保留={row['variance']}"
        )

    best = max(results, key=lambda item: item["accuracy"])
    print(f"\n🏆 最佳配置: {best['method']} | 测试准确率: {best['accuracy']:.2f}%")
    print_class_stats(best["class_stats"])

    results_dir = Path(args.results_dir)
    pd.DataFrame(rows).to_csv(results_dir / "bayes_pca_results.csv", index=False, encoding="utf-8-sig")
    save_json(results_dir / "bayes_pca_results.json", {"results": results})
    save_accuracy_plot(rows, args.figures_dir)

    print(f"\n💾 数值结果已保存至: {results_dir}")
    print(f"🖼️ 可视化图片已保存至: {args.figures_dir}")
    print("🎉 运行完成！")


if __name__ == "__main__":
    main()
