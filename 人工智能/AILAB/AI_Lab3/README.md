# 实验三：Fashion-MNIST 子集分类

比较 GaussianNB、PCA + GaussianNB、BernoulliNB、LeNet-5 和 ResidualCNN，并保留数据增强消融实验。

## 运行

在本实验目录内安装依赖：

```powershell
python -m pip install -r requirements.txt
```

自行准备原实验使用的五类服饰图像子集（标签 0、3、5、6、7），按以下结构放置。仓库不包含原始数据与权重。

```text
data/
  train/
    labels.csv
    images/
  test/
    labels.csv
    images/
```

CSV 必须包含 `filename` 和 `label_id` 两列；`filename` 对应 `images/` 中的图像名。原实验训练集 8000 张、测试集 2000 张。数据分布和随机性会影响重新运行的结果。

```powershell
python baseline/bayes.py
python baseline/lenet.py --epochs 5 --batch_size 16 --lr 0.02 --optimizer adam
python src/bayes_pca.py
python src/bayes_bernoulli.py
python src/lenet_base.py
python src/lenet_plus.py
```

可通过 `--train_dir`、`--test_dir` 指向其他本机数据位置。改进代码默认读取本实验目录下的 `data/train` 和 `data/test`。程序生成的权重、图片和日志默认不进入仓库。

## 实验材料

- `baseline/`：原实验参考代码的独立副本，默认数据路径已适配整理后的目录。
- `src/`：四组改进实验及公共工具。
- `results/`：原实验按配置和类别汇总的 CSV、JSON 指标，不包含逐样本数据。
- [脱敏报告 PDF](report/report.pdf)与 [LaTeX 源码](report/report.tex)：移除了身份封面和原始截图、插图。源码在 `report/` 下使用 XeLaTeX 连续编译两次。

旧提交副本、学校模板、参考课件、终端截图、数据集、权重和编译日志不随本目录发布。
