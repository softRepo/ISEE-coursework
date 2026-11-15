"""模型包入口。"""

from models.improved_cnn import ImprovedCNN
from models.mini_resnet import MiniResNet
from models.rgb_mini_resnet import RGBMiniResNet
from models.simple_cnn import SimpleCNN

# 复合表情实验复用 RGBMiniResNet，只在训练/评估脚本中传入 num_classes=11。
CompoundRGBMiniResNet = RGBMiniResNet
# RAF-ML 多标签实验同样复用 RGBMiniResNet，只在训练/评估脚本中传入 num_classes=6。
RAFMLRGBMiniResNet = RGBMiniResNet

__all__ = ["SimpleCNN", "ImprovedCNN", "MiniResNet", "RGBMiniResNet", "CompoundRGBMiniResNet", "RAFMLRGBMiniResNet"]
