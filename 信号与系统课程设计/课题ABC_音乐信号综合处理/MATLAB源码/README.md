# 课题ABC：音乐信号综合处理

## 1. 项目简介

本项目使用 MATLAB 完成三个音乐信号处理任务：

- 课题A：音乐片段的回声、混响和频域分析。
- 课题B：音频采样、降采样、重构和误差分析。
- 课题C：音频加噪、滤波器设计和去噪效果分析。

三个课题共用 `common/` 中的音频读取、预处理、频谱分析、绘图保存和表格写出函数。输入音频片段已放在 `segments/` 中，运行脚本后结果保存到 `results/` 下的对应课题目录。

## 2. 提交目录结构

```text
MATLAB源码/
├─ run_all_ABC.m
├─ config_ABC_experiments.m
├─ README.md
├─ common/
├─ segments/
├─ topic_A/
├─ topic_B/
└─ topic_C/
```

其中：

- `common/`：保存三个课题共用的函数文件。
- `segments/`：保存已经裁剪好的输入音频片段。
- `topic_A/`：保存课题A主脚本。
- `topic_B/`：保存课题B主脚本。
- `topic_C/`：保存课题C主脚本。

## 3. 输入文件

主程序默认读取 `segments/` 中的三个 wav 文件：

```text
segments/A_fur_elise_segment.wav
segments/B_hotel_california_segment.wav
segments/C_faded_segment.wav
```

文件用途如下：

- `A_fur_elise_segment.wav`：用于课题A音乐音效和频域分析。
- `B_hotel_california_segment.wav`：用于课题B采样、降采样和重构分析。
- `C_faded_segment.wav`：用于课题C加噪、滤波和去噪分析。

## 4. 运行环境

建议使用 MATLAB R2020a 或更高版本。涉及频谱分析、滤波和重采样的部分建议安装 Signal Processing Toolbox。

## 5. 运行方式

在 MATLAB 中进入本目录后运行：

```matlab
run_all_ABC
```

也可以分别运行三个课题主脚本：

```matlab
addpath('common', 'topic_A', 'topic_B', 'topic_C');
main_A_music_effect_analysis
main_B_sampling_reconstruction
main_C_noise_filter_design
```

运行前应确认 `segments/` 中三个输入 wav 文件存在。

## 6. 参数配置

`config_ABC_experiments.m` 集中设置三个课题的运行参数，包括目标采样率、是否显示图像、是否播放音频、回声延迟、混响增益、STFT 窗长、有效带宽能量比例、采样率设计系数、噪声强度、低通滤波器截止频率、滤波器阶数和陷波器带宽等。

常用参数示例：

```matlab
cfg.targetFs = 44100;
cfg.enablePlayback = false;
cfg.A.echoDelaySec = 0.35;
cfg.B.energyRatioForBandwidth = 0.99;
cfg.C.whiteNoiseSNRdB = 12;
cfg.C.noiseFreqHz = 3000;
```

## 7. 主要脚本

- `run_all_ABC.m`：总入口，添加公共函数路径，加载配置，依次执行课题A、B、C。
- `topic_A/main_A_music_effect_analysis.m`：完成钢琴片段的时域、FFT、STFT、回声和混响分析。
- `topic_B/main_B_sampling_reconstruction.m`：完成有效频率估计、不同采样率下的重采样和直接抽取，并计算重构误差。
- `topic_C/main_C_noise_filter_design.m`：完成白噪声和单频噪声叠加，设计低通和陷波滤波器，并比较去噪效果。

## 8. 公共函数作用

`common/` 中的函数主要用于：

- 音频读取与格式检查：`load_audio_segment.m`、`read_audio_robust.m`。
- 音频预处理与归一化：`preprocess_audio_basic.m`、`normalize_audio_peak.m`。
- 频谱与时频分析：`compute_fft_spectrum.m`、`compute_stft_manual.m`、`estimate_effective_bandwidth.m`。
- 指标计算：`audio_metrics.m`。
- 绘图与结果保存：`plot_audio_overview.m`、`plot_audio_spectrum.m`、`plot_time_frequency_overview.m`、`save_figure_safely.m`、`write_table_utf8.m`。
- 其他辅助处理：`extract_audio_segment.m`、`find_audio_by_keywords.m`、`safe_sound_play.m`。

## 9. 输出结果

运行后主要结果保存到以下目录：

```text
results/A_music_effect/
results/B_sampling/
results/C_denoising/
```

各课题目录下保存的内容如下：

- `results/A_music_effect/figures`：波形、频谱、STFT 和音效对比图。
- `results/A_music_effect/tables`：音效处理指标表。
- `results/A_music_effect/audio`：原始片段、回声片段和混响片段。
- `results/B_sampling/figures`：有效带宽、采样对比、频谱对比、混叠现象和误差图。
- `results/B_sampling/tables`：采样重构指标表。
- `results/B_sampling/audio`：不同采样条件下的代表性重构音频。
- `results/C_denoising/figures`：加噪对比、滤波器响应、去噪波形、频谱和参数对比图。
- `results/C_denoising/tables`：滤波参数和去噪指标表。
- `results/C_denoising/audio`：加噪音频和滤波后音频。

## 10. 常见问题

1. 找不到音频片段  
   检查 `segments/` 中三个 wav 文件是否存在，文件名应与配置保持一致。

2. 运行时间较长  
   课题B涉及重采样和重构，课题C涉及多组滤波参数比较，运行时间会随音频长度和参数组合增加。

3. 图像窗口较多  
   可在 `config_ABC_experiments.m` 中将 `cfg.showFigures` 设置为 `false`，只保存结果图像。

4. 播放音频不符合预期  
   默认不播放音频。如需听感检查，可在配置文件中将 `cfg.enablePlayback` 设置为 `true`。
