# 课题E：基于 MFCC 与 DTW 的孤立数字语音识别

## 1. 项目简介

本项目基于 Free Spoken Digit Dataset（FSDD）数据集，使用 MATLAB 实现 0-9 十个英文数字的孤立词识别。主要流程包括语音读取、预处理、分帧加窗、MFCC 特征提取、DTW 距离计算和模板匹配分类。

## 2. 提交目录结构

```text
MATLAB源码/
├─ main_topicE_dtw_speech_recognition.m
├─ config_topicE.m
├─ README.md
├─ dataset/
│  └─ recordings/
└─ functions/
   ├─ classify_by_dtw.m
   ├─ compute_delta_features.m
   ├─ compute_mfcc_manual.m
   ├─ dct2_manual.m
   ├─ dtw_distance_manual.m
   ├─ endpoint_detect_energy_zcr.m
   ├─ frame_signal.m
   ├─ hz2mel_local.m
   ├─ load_audio_mono.m
   ├─ mel2hz_local.m
   ├─ mel_filterbank.m
   ├─ normalize_audio.m
   ├─ parse_fsdd_filename.m
   ├─ plot_confusion_matrix_local.m
   ├─ plot_dtw_example.m
   ├─ preprocess_speech.m
   ├─ save_figure_safely.m
   └─ scan_fsdd_dataset.m
```

## 3. 输入文件

先按 [数据说明](dataset/README.md)从 FSDD 原仓库下载录音。语音数据位于：

```text
dataset/recordings/
```

音频文件名格式为：

```text
{digitLabel}_{speakerName}_{index}.wav
```

示例：

```text
0_george_0.wav
7_jackson_32.wav
```

程序从文件名解析数字标签、说话人名称和录音编号。默认数据划分方式为：

- 编号 `0-4`：测试样本。
- 编号 `5-49`：训练候选样本。

## 4. 运行环境

建议使用 MATLAB R2020a 或更高版本，并安装 Signal Processing Toolbox。MFCC 和 DTW 的核心计算由 `functions/` 中的函数完成。

## 5. 运行方式

在 MATLAB 中进入本目录后运行：

```matlab
main_topicE_dtw_speech_recognition
```

运行前应确认 `config_topicE.m` 中的数据路径指向实际的 `dataset/recordings/` 目录。

## 6. 参数配置

主要参数位于 `config_topicE.m`，包括：

- 识别数字范围：`0-9`。
- 测试编号：`0-4`。
- 训练编号：`5-49`。
- 每类最大模板数：`cfg.maxTemplatesPerDigit`。
- 模板选择方式：`cfg.templateSelectionMode`。
- 类别决策方式：`cfg.classDecisionMode`。
- 目标采样率：`8000 Hz`。
- 帧长：`25 ms`。
- 帧移：`10 ms`。
- NFFT：`256`。
- Mel 滤波器数量：`26`。
- MFCC 维数：`13`。
- 是否使用 C0 系数：`cfg.useC0`。
- 是否使用一阶差分特征：`cfg.useDelta`。
- 是否使用二阶差分特征：`cfg.useDeltaDelta`。
- DTW 距离类型：`cfg.dtwDistanceMetric`。
- 图像显示方式：`cfg.figureVisible`。
- 图像保存分辨率：`cfg.figureDpi`。

修改配置参数后，重新运行主脚本即可按新参数保存结果。

## 7. 主要脚本和函数

- `main_topicE_dtw_speech_recognition.m`：主入口，加载配置，扫描数据集，划分训练与测试样本，提取模板特征，执行 DTW 分类，统计识别结果，并保存图像和表格。
- `config_topicE.m`：集中保存数据路径、MFCC 参数、DTW 参数、模板数量、图像保存方式和输出目录配置。
- `scan_fsdd_dataset.m`：扫描 FSDD 数据集目录，整理 wav 文件路径、标签、说话人和编号信息。
- `parse_fsdd_filename.m`：解析 FSDD 文件名。
- `load_audio_mono.m`：读取音频文件，转换为单声道并返回采样率。
- `preprocess_speech.m`：完成去直流、归一化、预加重和有效语音片段提取。
- `endpoint_detect_energy_zcr.m`：基于短时能量和过零率进行端点检测。
- `frame_signal.m`：对语音信号进行分帧处理。
- `compute_mfcc_manual.m`：完成 MFCC 特征提取。
- `mel_filterbank.m`、`hz2mel_local.m`、`mel2hz_local.m`、`dct2_manual.m`：完成 Mel 频率映射和倒谱相关计算。
- `compute_delta_features.m`：计算一阶或二阶差分特征。
- `dtw_distance_manual.m`：计算 DTW 距离并回溯最优路径。
- `classify_by_dtw.m`：根据测试样本与模板的 DTW 距离完成类别判决。
- `plot_confusion_matrix_local.m`、`plot_dtw_example.m`、`save_figure_safely.m`：绘制并保存识别结果图像。

## 8. 输出结果

运行后主要结果保存到 `results/` 下：

- `results/figures`：保存数据统计图、预处理示例图、MFCC热力图、DTW路径图、混淆矩阵和类别准确率图。
- `results/tables`：保存预测结果、每类准确率和混淆矩阵表格。

关键表格文件包括：

- `prediction_results.csv`：逐条测试样本的真实标签、预测标签、说话人、文件名、DTW 距离和识别结果。
- `class_accuracy.csv`：每个数字类别的测试样本数、正确识别数和类别准确率。
- `confusion_matrix.csv`：0-9 十类数字识别的混淆矩阵。

## 9. 常见问题

1. 找不到数据集  
   检查 `dataset/recordings/` 是否存在，并确认其中包含 FSDD 的 wav 文件。

2. 数据集路径不一致  
   可在 `config_topicE.m` 中修改数据集路径参数。

3. 运行时间较长  
   DTW 需要逐一比较测试样本和模板样本，运行时间会随模板数量和测试数量增加。可适当减小 `cfg.maxTemplatesPerDigit` 或 `cfg.maxTestsPerDigit`。

4. 准确率发生变化  
   修改模板数量、MFCC 参数、差分特征或 DTW 距离方式后，识别准确率可能变化。

5. 图像窗口较多  
   可在 `config_topicE.m` 中将 `cfg.figureVisible` 设置为 `'off'`，只保存结果图像。
