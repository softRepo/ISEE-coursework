# 实验三：Fashion-MNIST 子集分类

比较 GaussianNB、PCA + GaussianNB、BernoulliNB、LeNet-5 和 ResidualCNN，并进行数据增强消融实验。

## 运行

```powershell
python -m pip install -r requirements.txt
```

准备标签为 0、3、5、6、7 的 Fashion-MNIST 五类子集，按以下目录放置：

```text
data/
  train/
    labels.csv
    images/
  test/
    labels.csv
    images/
```

CSV 包含 `filename` 和 `label_id` 两列，图像文件放在对应的 `images/` 下。实验使用 8000 张训练图像和 2000 张测试图像。

```powershell
python baseline/bayes.py
python baseline/lenet.py --epochs 5 --batch_size 16 --lr 0.02 --optimizer adam
python src/bayes_pca.py
python src/bayes_bernoulli.py
python src/lenet_base.py
python src/lenet_plus.py
```

默认读取本实验目录下的 `data/train` 和 `data/test`，也可通过 `--train_dir`、`--test_dir` 指定数据位置。

## 文件说明

- `baseline/`：基础 GaussianNB 和 LeNet 实现。
- `src/`：改进实验及公共工具。
- `results/`：CSV、JSON 实验指标。
- [实验报告 PDF](report/report.pdf)与 [LaTeX 源码](report/report.tex)。源码在 `report/` 下使用 XeLaTeX 连续编译两次。
