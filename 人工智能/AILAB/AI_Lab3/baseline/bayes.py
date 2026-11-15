from pathlib import Path
import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# ==========================================
# 1. 数据集加载函数
# ==========================================
def load_data(data_dir):
    csv_file = os.path.join(data_dir, 'labels.csv')
    img_dir = os.path.join(data_dir, 'images')
    df = pd.read_csv(csv_file)

    X = []
    y = []
    for idx, row in df.iterrows():
        img_name = row['filename']
        label = row['label_id']
        img_path = os.path.join(img_dir, img_name)
        image = Image.open(img_path).convert('L')
        img_flat = np.array(image).flatten()
        X.append(img_flat)
        y.append(label)
    return np.array(X), np.array(y)

# ==========================================
# 2. 参数解析器
# ==========================================
def get_args_parser():
    parser = argparse.ArgumentParser(description='原生高斯朴素贝叶斯 - FashionMNIST子集')
    # 双路径参数（和LeNet完全一致）
    parser.add_argument('--train_dir', type=str, default=str(Path(__file__).resolve().parents[1] / 'data' / 'train'), help='训练集路径')
    parser.add_argument('--test_dir', type=str, default=str(Path(__file__).resolve().parents[1] / 'data' / 'test'), help='测试集路径')
    # 基础超参数
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    return parser

# 类别映射（和LeNet一致）
LABEL_NAME_MAP = {
    0: "T-shirt/top",
    3: "Dress",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker"
}
TARGET_CLASSES = [0, 3, 5, 6, 7]

# ==========================================
# 3. 主程序
# ==========================================
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    print("🚀 启动 原生高斯朴素贝叶斯 训练")
    print(f"📂 训练集路径: {args.train_dir}")
    print(f"📂 测试集路径: {args.test_dir}")

    # 加载数据
    X_train, y_train = load_data(args.train_dir)
    X_test, y_test = load_data(args.test_dir)

    print(f"📊 训练集: {len(X_train)} 张 | 测试集: {len(X_test)} 张")

    # --------------------------
    # 原生高斯朴素贝叶斯
    # --------------------------

    # 创建model对象
    model = GaussianNB()
    # X_train是训练集图片特征、y_train是训练集真实标签
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    overall_acc = accuracy_score(y_test, y_pred) * 100

    # 统计每个类别的准确率
    class_correct = {c: 0 for c in TARGET_CLASSES}
    class_total = {c: 0 for c in TARGET_CLASSES}

    for true, pred in zip(y_test, y_pred):
        if true in TARGET_CLASSES:
            class_total[true] += 1
            if true == pred:
                class_correct[true] += 1

    # --------------------------
    # 打印结果（和LeNet格式相同）
    # --------------------------
    print("\n" + "="*50)
    print(f"🎯 原生朴素贝叶斯 测试结果")
    print("="*50)
    print(f"✅ 总体测试准确率: {overall_acc:.2f}%")
    print("\n📊 各类别正确率：")
    for c in TARGET_CLASSES:
        acc = 100 * class_correct[c] / class_total[c] if class_total[c] != 0 else 0
        print(f"   类别{c:2d} ({LABEL_NAME_MAP[c]:10s}): {class_correct[c]}/{class_total[c]} | 准确率={acc:.2f}%")

    print("\n🎉 运行完成！")
