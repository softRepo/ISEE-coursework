from pathlib import Path
import os
import time
import argparse
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import psutil

# ==========================================
# 1. 自定义数据集类
# ==========================================
class FashionMiniDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        参数:
            data_dir (string): 数据集主目录，内部应包含 'labels.csv' 和 'images/' 文件夹。
            transform (callable, optional): 图像预处理操作。
        """
        self.data_dir = data_dir
        self.csv_file = os.path.join(data_dir, 'labels.csv')
        self.img_dir = os.path.join(data_dir, 'images')

        # 读取 csv 文件
        self.data_frame = pd.read_csv(self.csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 根据 csv 中的记录获取图片名和标签
        img_name = self.data_frame.iloc[idx, 0]
        label = self.data_frame.iloc[idx, 1]

        # 读取图片转为单通道灰度图
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label

# ==========================================
# 2. 定义经典的 LeNet-5 网络结构
# ==========================================
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# 3. 参数解析器 (Argparse)
# ==========================================
def get_args_parser():
    parser = argparse.ArgumentParser(description='LeNet-5 Fashion-MNIST CPU Training')

    # 路径配置
    parser.add_argument('--train_dir', type=str, default=str(Path(__file__).resolve().parents[1] / 'data' / 'train'),
                        help='Training dataset directory')
    parser.add_argument('--test_dir', type=str, default=str(Path(__file__).resolve().parents[1] / 'data' / 'test'),
                        help='Testing dataset directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Log directory (env: TRAIN_LOG_PATH) to save results')
    # 超参数配置
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Input batch size for training')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop'], help='select optimizer: adam, sgd, rmsprop')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')

    return parser

# ==========================================
# 4. 主函数
# ==========================================
if __name__ == '__main__':
    # 获取解析后的参数
    parser = get_args_parser()
    args = parser.parse_args()

    # 强制指定设备为 CPU
    device = torch.device('cpu')
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 启动训练 | 设备: {device}")

    # 如果指定了日志目录，则创建它
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        print(f"📁 日志及输出将保存在: {args.log_dir}")

    # 1. 配置数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 2. 分别加载训练集和测试集
    try:
        train_dataset = FashionMiniDataset(data_dir=args.train_dir, transform=transform)
        test_dataset = FashionMiniDataset(data_dir=args.test_dir, transform=transform)
    except FileNotFoundError as e:
        print(f"\n❌ 错误: 找不到数据集目录或文件。请确保路径结构正确！\n详细信息: {e}")
        exit(1)

    print(f"📊 加载成功 | 训练集: {len(train_dataset)} 张 | 测试集: {len(test_dataset)} 张")

    # 3. 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. 初始化模型、损失函数和优化器
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    # 5. 训练循环
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{args.epochs}]", unit="batch")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 实时更新进度条后缀（动态显示当前 Loss 和 CPU 使用率）
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cpu': f"{psutil.cpu_percent()}%"
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"✅ Epoch[{epoch+1:02d}] 结束 | 平均 Loss: {epoch_loss:.4f} | 本轮耗时: {epoch_time:.2f}s\n")

    # 6. 测试集评估
    print("\n🔍 开始评估测试集...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"🎉 训练完成！测试集最终准确率: {accuracy:.2f}%")

    # 7. (可选) 保存日志或模型
    if args.log_dir is not None:
        log_file = os.path.join(args.log_dir, 'training_result.txt')
        with open(log_file, 'w') as f:
            f.write(f"Learning Rate: {args.lr}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Final Accuracy: {accuracy:.2f}%\n")

        # 也可以保存训练好的模型权重
        model_path = os.path.join(args.log_dir, 'lenet_cpu.pth')
        torch.save(model.state_dict(), model_path)
        print(f"💾 结果已保存至 {args.log_dir}")
