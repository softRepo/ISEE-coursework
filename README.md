# 人工智能实验与课程项目

本仓库将课程实验和人脸表情识别大作业统一整理在 [人工智能](人工智能/) 目录，发布经过隐私处理的源码、实验结果和文档。

```text
人工智能/
├── AILAB/
│   ├── AI_Lab1/        # 八数码与 A* 搜索
│   ├── AI_Lab2/        # 对弈搜索与 Alpha–Beta 剪枝
│   └── AI_Lab3/        # Fashion-MNIST 分类
└── 人脸表情识别/        # PyTorch + PyQt5 课程大作业
```

| 内容 | 入口 | 实验材料 |
| --- | --- | --- |
| AILAB 三项课程实验 | [说明与运行方法](人工智能/AILAB/README.md) | 各实验源码、脱敏报告；实验三含汇总指标和 LaTeX 源码 |
| 人脸表情识别项目 | [项目说明](人工智能/人脸表情识别/README.md) | [报告 PDF](人工智能/人脸表情识别/docs/report.pdf) · [演示 PPTX](人工智能/人脸表情识别/docs/presentation.pptx) |

请进入各项目目录，按各自 README 安装依赖和运行。原有项目入口与文档现已移动到 `人工智能/人脸表情识别/`；仓库地址与已有提交历史保留。

原始数据集、模型权重、截图、原始提交包、带身份信息的文档和本机环境不上传。脱敏范围与检查方式见 [PRIVACY.md](PRIVACY.md)。仓库保持私有。

```powershell
# 在仓库根目录检查暂存内容。
git add .
python tools/check_privacy.py --tracked
git config core.hooksPath .githooks
```
