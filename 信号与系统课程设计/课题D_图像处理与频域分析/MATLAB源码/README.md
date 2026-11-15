# 课题D：图像处理与频域分析

## 1. 项目简介

本项目使用 MATLAB 对给定图像进行二维频域分析。程序读取输入图像，转换为灰度图像，计算二维傅里叶频谱，并构造二维高斯低通滤波器和二维高斯高通滤波器，对不同 `sigma` 参数下的低通和高通结果进行比较。

## 2. 提交目录结构

```text
MATLAB源码/
├─ main_topicD_frequency_filter.m
├─ README.md
├─ 课题D给定图像.png
└─ functions/
   ├─ apply_frequency_filter.m
   ├─ find_given_image.m
   ├─ gaussian_frequency_filters.m
   ├─ normalize_for_display.m
   ├─ read_and_preprocess_image.m
   ├─ save_figure_safely.m
   └─ show_filtering_results.m
```

## 3. 输入文件

程序默认在当前目录查找主文件名为 `课题D给定图像` 的图像文件。提交目录中包含：

```text
课题D给定图像.png
```

常见图像格式包括 `.png`、`.jpg`、`.jpeg`、`.bmp`、`.tif` 和 `.tiff`。

## 4. 运行环境

建议使用 MATLAB R2020a 或更高版本。图像读取、灰度转换和图像保存部分可使用 Image Processing Toolbox；部分函数提供了基础兼容处理。

## 5. 运行方式

在 MATLAB 中进入本目录后运行：

```matlab
main_topicD_frequency_filter
```

主脚本会定位自身目录，添加 `functions/` 路径，读取输入图像，并保存频域滤波结果。

## 6. 参数配置

主要参数位于 `main_topicD_frequency_filter.m` 开头：

```matlab
sigma_list = [10, 30, 80];
mesh_downsample = 4;
save_figures = true;
show_figures = true;
default_sigma = 30;
```

参数含义：

- `sigma_list`：需要比较的高斯滤波器参数。
- `mesh_downsample`：三维滤波器图显示时的下采样倍数，不影响实际计算。
- `save_figures`：是否保存结果图像。
- `show_figures`：是否显示 MATLAB 图窗。
- `default_sigma`：综合展示图使用的默认参数。

## 7. 主要脚本和函数

- `main_topicD_frequency_filter.m`：主入口，完成图像读取、灰度转换、频谱计算、高斯滤波器构造、频域滤波、结果绘制和参数表保存。
- `find_given_image.m`：查找给定图像文件。
- `read_and_preprocess_image.m`：读取图像并完成灰度转换、类型转换和归一化处理。
- `gaussian_frequency_filters.m`：根据图像尺寸和 `sigma` 构造高斯低通、高通滤波器。
- `apply_frequency_filter.m`：完成频域相乘、逆中心化和二维逆傅里叶变换。
- `show_filtering_results.m`：绘制原图、频谱、滤波器、滤波结果和不同参数对比图。
- `normalize_for_display.m`：将数据归一化到适合显示和保存的范围。
- `save_figure_safely.m`：保存图像文件，并兼容不同 MATLAB 版本的图像导出方式。

## 8. 输出结果

运行后主要结果保存到 `results/` 下：

```text
results/images/
results/figures/
results/tables/
```

目录用途如下：

- `results/images`：保存单张结果图像，例如原始灰度图、频谱图、低通结果、高通结果和滤波器图。
- `results/figures`：保存综合对比图和参数对比图。
- `results/tables`：保存参数和统计结果表格。

关键输出文件包括：

```text
original_gray.png
original_spectrum_log.png
lowpass_result_sigma_10.png
lowpass_result_sigma_30.png
lowpass_result_sigma_80.png
highpass_result_sigma_10.png
highpass_result_sigma_30.png
highpass_result_sigma_80.png
gaussian_lowpass_filter_sigma_10.png
gaussian_lowpass_filter_sigma_30.png
gaussian_lowpass_filter_sigma_80.png
gaussian_highpass_filter_sigma_10.png
gaussian_highpass_filter_sigma_30.png
gaussian_highpass_filter_sigma_80.png
filter_parameter_record.csv
```

`filter_parameter_record.csv` 保存不同 `sigma` 参数下的低通结果均值、低通结果标准差、高通结果均值和高通结果标准差。

## 9. 常见问题

1. 找不到给定图像  
   检查当前目录下是否存在主文件名为 `课题D给定图像` 的图像文件。

2. 图像显示过暗或过亮  
   程序会对灰度图、频谱图和高通结果进行适合显示的归一化处理。

3. 修改 `sigma` 后结果变化明显  
   `sigma` 直接影响频域保留范围。较小的 `sigma` 会增强低通平滑和高通边缘提取效果，较大的 `sigma` 会保留更多细节。
