# 消融实验与对比分析汇总

## 输入形式消融

| experiment_name | dataset | task_type | input_type | training_strategy | accuracy | micro_f1 | macro_precision | macro_recall | macro_f1 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FER2013 灰度 MiniResNet 在 RAF-DB 外部测试 | RAF-DB test | 7类softmax | 灰度48×48 | 外部测试 | 0.5684 |  | 0.4885 | 0.4187 | 0.3921 | FER2013 灰度模型跨数据集测试，存在明显域差异。 |
| RAF-DB RGB MiniResNet | RAF-DB test | 7类softmax | RGB96×96 | RAF-DB 从头训练 | 0.8380 |  | 0.7597 | 0.7621 | 0.7591 | 同源 RAF-DB 彩色图像训练与测试，保留颜色和纹理信息。 |

输入形式消融结果表明，FER2013 灰度 MiniResNet 在 RAF-DB 外部测试上的表现明显低于 RAF-DB RGB MiniResNet。前者同时受到灰度输入、低分辨率 FER2013 风格和跨数据集域差异影响；后者使用 RAF-DB 同源彩色真实场景图像训练，RGB 输入保留了更多颜色、纹理和光照信息，因此 Accuracy 和 Macro F1 均有显著提升。

## 训练策略消融

| experiment_name | dataset | task_type | input_type | training_strategy | accuracy | micro_f1 | macro_precision | macro_recall | macro_f1 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAF-DB RGB 从头训练 | RAF-DB test | 7类softmax | RGB96×96 | 从头训练 | 0.8380 |  | 0.7597 | 0.7621 | 0.7591 | 使用 RAF-DB 训练集从头训练，测试集与训练集同源。 |
| AffectNet 预训练 + RAF-DB 微调 | RAF-DB test | 7类softmax | RGB96×96 | 预训练微调 | 0.8181 |  | 0.7476 | 0.7179 | 0.7302 | 先在 AffectNet 上预训练，再迁移到 RAF-DB 上微调。 |

训练策略消融结果显示，AffectNet 预训练 + RAF-DB 微调取得了接近 RAF-DB 从头训练的性能，但在 RAF-DB test 上仍略低于同源数据从头训练模型。这说明预训练具有一定迁移价值，但 AffectNet 与 RAF-DB 在数据分布、图像风格、标签映射和类别比例上的差异可能限制了迁移效果。

## 任务形式对比

| experiment_name | dataset | task_type | input_type | training_strategy | accuracy | micro_f1 | macro_precision | macro_recall | macro_f1 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAF-DB RGB 基本七类表情识别 | RAF-DB test | 7类softmax | RGB96×96 | RAF-DB 从头训练 | 0.8380 |  | 0.7597 | 0.7621 | 0.7591 | 基本七类表情识别任务，类别定义相对稳定。 |
| RAF-DB Compound 11类复合表情识别 | RAF-DB Compound test | 11类softmax | RGB96×96 | AffectNet 预训练微调 | 0.5846 |  | 0.4630 | 0.4435 | 0.4492 | 复合表情类别更细，类别语义相近，样本数量更少。 |
| RAF-ML 6维多标签混合表情识别 | RAF-ML test | 6维sigmoid多标签 | RGB96×96 | AffectNet 预训练微调 |  | 0.7588 | 0.7592 | 0.7163 | 0.7352 | 多标签 sigmoid 任务，没有普通单标签 Accuracy，主指标为 Micro F1 和 Macro F1。 |

任务形式对比结果表明，基本七类表情识别相对更稳定；RAF-DB Compound 11 类复合表情识别难度更高，主要原因是类别更细、语义更接近且样本数量较少；RAF-ML 采用 6 维 sigmoid 多标签形式表达混合情绪，Macro F1 较高，说明多标签形式更适合描述真实场景中多个情绪成分同时存在的情况。
