# 基于 PyQt5 与深度学习的人脸表情识别系统设计与实现

实验报告：[阅读脱敏 PDF（34 页）](docs/report.pdf) · [LaTeX 源码](docs/report.tex)

## 1. 项目简介

本项目基于 PyTorch 和 PyQt5 实现人脸表情识别系统。系统支持基本七类表情识别、复合表情识别、多标签混合表情识别，并提供图片上传识别和摄像头实时识别 GUI。

项目主要包含以下功能：

- FER2013 灰度表情识别模型训练与评估；
- CK+ 和 RAF-DB 外部泛化测试；
- RAF-DB RGB 表情识别模型训练与评估；
- AffectNet RGB Full 训练与 AffectNet 预训练 + RAF-DB 微调；
- RAF-DB Compound 11 类复合表情识别；
- RAF-ML 6 维 sigmoid 多标签混合表情识别；
- PyQt5 GUI 图片上传识别与摄像头实时识别；
- 消融实验结果整理脚本。

## 2. 运行环境

推荐运行环境如下：

- 操作系统：Windows
- Python 虚拟环境名称：`face_emotion_env`
- 深度学习框架：PyTorch
- GUI：PyQt5
- 图像处理：OpenCV、Pillow
- 数据处理：numpy、pandas
- 评估与可视化：scikit-learn、matplotlib

环境安装命令：

```powershell
python -m venv face_emotion_env
.\face_emotion_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如需使用 GPU，请根据本机 CUDA 版本到 PyTorch 官网选择对应安装命令。

## 3. 代码结构

源码提交包主要包含以下文件和目录：

```text
face_emotion_project/
├── models/
│   ├── detectors/
│   ├── simple_cnn.py
│   ├── improved_cnn.py
│   ├── mini_resnet.py
│   └── rgb_mini_resnet.py
├── utils/
│   ├── dataset.py
│   ├── rafdb_dataset.py
│   ├── affectnet_dataset.py
│   ├── compound_dataset.py
│   ├── rafml_dataset.py
│   ├── metrics.py
│   ├── predict_utils.py
│   └── visualize.py
├── train.py
├── evaluate.py
├── ckplus_eval.py
├── rafdb_eval.py
├── train_rafdb_rgb.py
├── evaluate_rafdb_rgb.py
├── train_affectnet_rgb.py
├── evaluate_affectnet_rgb.py
├── finetune_rafdb_from_affectnet.py
├── train_compound_rgb.py
├── evaluate_compound_rgb.py
├── train_rafml_multilabel.py
├── evaluate_rafml_multilabel.py
├── analyze_ablation_results.py
├── gui.py
├── requirements.txt
└── README.md
```

`data/` 为本地数据集目录，源码提交包中不包含完整数据集。运行前需要按照本 README 的数据集目录说明自行放置数据。

## 4. 主要代码文件说明

| 文件 | 功能 |
| --- | --- |
| `models/simple_cnn.py` | 基础 CNN 模型，用于 FER2013 灰度基线实验。 |
| `models/improved_cnn.py` | 改进 CNN 模型，加入更深网络结构、BatchNorm 和 Dropout。 |
| `models/mini_resnet.py` | 灰度 MiniResNet，用于 FER2013 主模型和外部泛化测试。 |
| `models/rgb_mini_resnet.py` | RGBMiniResNet，用于 RAF-DB、AffectNet、Compound 和 RAF-ML。 |
| `models/detectors/` | 摄像头人脸定位模型目录，可放置 YuNet ONNX 和 MediaPipe tflite 模型。 |
| `utils/dataset.py` | FER2013 数据读取与检查。 |
| `utils/rafdb_dataset.py` | RAF-DB 数据读取与 RAF-DB 到 FER2013 的标签映射。 |
| `utils/affectnet_dataset.py` | AffectNet YOLO Format 数据读取与标签映射。 |
| `utils/compound_dataset.py` | RAF-DB Compound 复合表情数据读取。 |
| `utils/rafml_dataset.py` | RAF-ML 多标签混合表情数据读取。 |
| `utils/metrics.py` | 评估指标计算。 |
| `utils/predict_utils.py` | GUI 预测时使用的模型加载、图像预处理和推理工具。 |
| `utils/visualize.py` | 混淆矩阵、曲线和样例可视化工具。 |
| `train.py` | FER2013 灰度模型训练。 |
| `evaluate.py` | FER2013 灰度模型评估。 |
| `ckplus_eval.py` | CK+ 外部泛化测试。 |
| `rafdb_eval.py` | RAF-DB 灰度外部测试。 |
| `train_rafdb_rgb.py` | RAF-DB RGB 模型训练。 |
| `evaluate_rafdb_rgb.py` | RAF-DB RGB 模型评估。 |
| `train_affectnet_rgb.py` | AffectNet RGB 模型训练。 |
| `evaluate_affectnet_rgb.py` | AffectNet RGB 模型评估。 |
| `finetune_rafdb_from_affectnet.py` | 加载 AffectNet 预训练权重后在 RAF-DB 上微调。 |
| `train_compound_rgb.py` | RAF-DB Compound 11 类复合表情训练。 |
| `evaluate_compound_rgb.py` | RAF-DB Compound 模型评估。 |
| `train_rafml_multilabel.py` | RAF-ML 6 维多标签混合表情训练。 |
| `evaluate_rafml_multilabel.py` | RAF-ML 多标签模型评估。 |
| `analyze_ablation_results.py` | 消融实验结果整理，生成汇总表和对比图。 |
| `gui.py` | PyQt5 图形界面，支持图片上传识别和摄像头实时识别。 |

## 5. 数据集目录说明

运行代码前，请将数据集放置到以下路径：

```text
data/fer2013/
data/ck_dataset/
data/RAF-DB/DATASET/
data/AffectNet/YOLO_format/
data/compound/
data/RAF-ML/
```

说明：

- `data/fer2013/`：FER2013 ImageFolder 数据，包含 `train/`、`validation/` 或 `val/`、`test/`；
- `data/ck_dataset/ckextended.csv`：CK+ 表格数据，包含 `emotion`、`pixels`、`Usage` 三列；
- `data/RAF-DB/DATASET/`：RAF-DB Basic 数据，包含 `train/1..7/` 和 `test/1..7/`；
- `data/AffectNet/YOLO_format/`：AffectNet YOLO Format 数据，包含 `train/images`、`train/labels`、`valid/images`、`valid/labels`、`test/images`、`test/labels`；
- `data/compound/`：RAF-DB Compound 数据，包含 `Image/aligned/` 和 `EmoLabel/`；
- `data/RAF-ML/`：RAF-ML 数据，包含 `Image/aligned/` 和 `EmoLabel/`。

由于数据集体积较大，源码提交包中不包含完整数据集，运行前需按上述结构自行放置。

## 6. 主要运行命令

检查 FER2013 数据：

```powershell
python -m utils.dataset --data_dir data/fer2013
```

训练 FER2013 MiniResNet：

```powershell
python train.py --model mini_resnet --epochs 50 --batch_size 64 --lr 0.001 --data_dir data/fer2013 --use_aug --num_workers 0
```

评估 FER2013 MiniResNet：

```powershell
python evaluate.py --model mini_resnet --weights checkpoints/best_mini_resnet.pth --data_dir data/fer2013 --batch_size 64 --num_workers 0
```

CK+ 外部测试：

```powershell
python ckplus_eval.py --model mini_resnet --weights checkpoints/best_mini_resnet.pth --batch_size 64 --num_workers 0
```

RAF-DB 灰度外部测试：

```powershell
python rafdb_eval.py --model mini_resnet --weights checkpoints/best_mini_resnet.pth --data_dir data/RAF-DB/DATASET --split test --batch_size 64 --num_workers 0
```

RAF-DB RGB 训练：

```powershell
python train_rafdb_rgb.py --model rgb_mini_resnet --data_dir data/RAF-DB/DATASET --epochs 50 --batch_size 64 --lr 0.001 --num_workers 0 --use_aug
```

RAF-DB RGB 评估：

```powershell
python evaluate_rafdb_rgb.py --model rgb_mini_resnet --weights checkpoints/best_rafdb_rgb_mini_resnet.pth --data_dir data/RAF-DB/DATASET --batch_size 64 --num_workers 0
```

AffectNet RGB Full 训练：

```powershell
python train_affectnet_rgb.py --data_dir data/AffectNet/YOLO_format --epochs 40 --batch_size 128 --lr 0.001 --num_workers 0 --use_aug --use_all_data --tag affectnet_rgb_full_40epoch
```

AffectNet RGB Full 评估：

```powershell
python evaluate_affectnet_rgb.py --model rgb_mini_resnet --weights checkpoints/best_affectnet_rgb_full_40epoch.pth --data_dir data/AffectNet/YOLO_format --batch_size 128 --num_workers 0 --tag affectnet_rgb_full_40epoch
```

AffectNet 预训练 + RAF-DB 微调：

```powershell
python finetune_rafdb_from_affectnet.py --rafdb_dir data/RAF-DB/DATASET --pretrained checkpoints/best_affectnet_rgb_full_40epoch.pth --epochs 50 --batch_size 64 --lr 0.0001 --num_workers 0 --use_aug
```

RAF-DB Compound 训练：

```powershell
python train_compound_rgb.py --data_dir data/compound --epochs 40 --batch_size 64 --lr 0.0005 --num_workers 0 --use_aug --pretrained_weights checkpoints/best_affectnet_rgb_full_40epoch.pth --tag compound_rgb_affectnet_pretrained
```

RAF-DB Compound 评估：

```powershell
python evaluate_compound_rgb.py --data_dir data/compound --weights checkpoints/best_compound_rgb_affectnet_pretrained.pth --batch_size 64 --num_workers 0 --tag compound_rgb_affectnet_pretrained
```

RAF-ML 多标签训练：

```powershell
python train_rafml_multilabel.py --data_dir data/RAF-ML --epochs 50 --batch_size 64 --lr 0.0005 --num_workers 0 --use_aug --pretrained_weights checkpoints/best_affectnet_rgb_full_40epoch.pth --tag rafml_affectnet_pretrained
```

RAF-ML 多标签评估：

```powershell
python evaluate_rafml_multilabel.py --data_dir data/RAF-ML --weights checkpoints/best_rafml_affectnet_pretrained.pth --batch_size 64 --num_workers 0 --tag rafml_affectnet_pretrained
```

运行 GUI：

```powershell
python gui.py
```

消融实验整理：

```powershell
python analyze_ablation_results.py
```

## 7. GUI 使用说明

GUI 入口为：

```powershell
python gui.py
```

GUI 支持：

- 图片上传识别；
- 摄像头实时识别；
- 基本表情识别、复合表情识别、混合表情识别三类任务；
- softmax 单标签输出和 sigmoid 多标签输出；
- YuNet、MediaPipe 和中心区域三种摄像头人脸定位方式；
- 根据任务类型自动切换类别标签、概率条数量和预测结果显示方式。

人脸定位方式仅在摄像头实时识别模式下可选；图片上传识别模式下会显示“当前模式不可用”。

摄像头检测模型默认路径：

```text
models/detectors/face_detection_yunet_2023mar.onnx
models/detectors/blaze_face_short_range.tflite
```

如果 YuNet 或 MediaPipe 不可用，GUI 会自动回退到中心区域识别。

## 8. 模型权重与实验材料

本仓库发布训练、评估和 GUI 源码，以及经过检查的汇总实验结果。模型权重、第三方检测器二进制文件、原始数据集、逐样本人脸图像、摄像头截图和原始课程提交材料均不随仓库分发。

- 模型权重需要按上述训练命令自行生成，放入 `checkpoints/`；GUI 首次使用前需要准备所选模型的权重。
- 可选检测器的文件名和来源说明见 `models/detectors/README.md`。
- 脱敏实验报告可直接阅读 [PDF（34 页）](docs/report.pdf)，源码为 `docs/report.tex`，可使用 XeLaTeX 编译；原始插图已移除，位置保留为占位说明。发布的 PDF 已清除文档属性，并检查正文、书签及附件。
- 仅汇总的分类统计和消融分析位于 `docs/results/`，不包含逐样本数据或人脸图片。
- 评估脚本默认不导出样例图片。只有主动指定大于 0 的 `--max_samples`、`--max_wrong_samples`、`--max_correct_samples` 或 `--max_prediction_samples` 时，才会生成对应样例；导出的图片仍包含人脸，不应提交到仓库。
- GUI 的图片选择功能在本机读取图片，摄像头识别在本机处理；当前源码没有向外部服务器上传图像的逻辑。

隐私范围和检查方法见 [PRIVACY.md](PRIVACY.md)。提交前运行 `python tools/check_privacy.py --tracked`，可通过 `git config core.hooksPath .githooks` 启用本地提交检查。

## 9. 常见问题

1. **数据集路径不存在**  
   请检查 `data/` 下的数据目录是否与第 5 节一致。

2. **权重文件不存在**  
   请确认对应 `.pth` 文件位于 `checkpoints/`，或先运行训练命令生成权重。

3. **Windows 数据加载报错**  
   建议运行命令时使用 `--num_workers 0`。

4. **PyQt5 缺失**  
   请运行 `pip install PyQt5`。

5. **摄像头打不开**  
   请检查摄像头是否被其他软件占用，系统是否授予摄像头权限，并确认已安装 `opencv-python`。

6. **YuNet 或 MediaPipe 模型缺失**  
   请将对应模型文件放到 `models/detectors/`。

7. **OpenCV 不支持 `FaceDetectorYN`**  
   可以运行 `pip install --upgrade opencv-python`。

8. **摄像头检测不到人脸**  
   请调整光线、正对摄像头并适当靠近；检测器失败时 GUI 会回退到中心区域识别。
