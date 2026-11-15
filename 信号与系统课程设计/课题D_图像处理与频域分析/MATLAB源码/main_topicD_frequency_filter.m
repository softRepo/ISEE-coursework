%% 课题D：图像处理与频域分析
% 本脚本是课题D图像频域滤波实验的主入口，主要完成以下处理：
% 1. 查找并读取“课题D给定图像”；
% 2. 将彩色图像转换为灰度图像，并归一化到 [0,1]；
% 3. 按 fft2、fftshift、频域相乘、ifftshift、ifft2 的流程完成
%    二维高斯低通和高通频域滤波；
% 4. 保存单张结果图、综合对比图和参数记录表。

clear; clc; close all;

%% ======================= 参数配置区 =======================
sigma_list = [10, 30, 80];   % 频域高斯滤波器宽度，单位为频率索引像素
mesh_downsample = 4;         % 三维频谱绘图下采样倍数；滤波计算仍使用完整尺寸
save_figures = true;         % 是否保存 .png 和 .fig
show_figures = true;         % 是否显示 Figure 窗口

default_sigma = 30;          % Figure 2、Figure 3、Figure 6 使用的默认 sigma
given_image_stem = '课题D给定图像';

%% ======================= 定位项目路径 =======================
script_full_path = mfilename('fullpath');
if isempty(script_full_path)
    topic_dir = pwd;
else
    topic_dir = fileparts(script_full_path);
end

functions_dir = fullfile(topic_dir, 'functions');
results_dir = fullfile(topic_dir, 'results');
figures_dir = fullfile(results_dir, 'figures');
images_dir = fullfile(results_dir, 'images');
tables_dir = fullfile(results_dir, 'tables');
report_dir = fullfile(topic_dir, 'report');

if exist(functions_dir, 'dir') ~= 7
    error('未找到 functions 目录：%s。请确认课题D项目结构完整。', functions_dir);
end
addpath(functions_dir);

output_dirs = {results_dir, figures_dir, images_dir, tables_dir, report_dir};
for k = 1:numel(output_dirs)
    if exist(output_dirs{k}, 'dir') ~= 7
        mkdir(output_dirs{k});
    end
end

%% ======================= 定位并读取给定图像 =======================
image_path = find_given_image(topic_dir, given_image_stem);
if isempty(image_path)
    error(['未找到名为“%s”的图像文件。请将 %s.png/.jpg/.jpeg/.bmp/.tif/.tiff ', ...
           '等格式的图像放到课题D目录下：%s'], ...
           given_image_stem, given_image_stem, topic_dir);
end

[original_display, gray_image, image_info] = read_and_preprocess_image(image_path);
[M, N] = size(gray_image);

sigma_list = double(sigma_list(:)');
sigma_list = sigma_list(isfinite(sigma_list) & sigma_list > 0);
if isempty(sigma_list)
    error('sigma_list 中没有有效的正数参数，请至少设置一个正的 sigma。');
end
mesh_downsample = max(1, round(mesh_downsample));

fprintf('\n========== 课题D：图像处理与频域分析 ==========\n');
fprintf('当前课题D目录：%s\n', topic_dir);
fprintf('找到给定图像：%s\n', image_path);
fprintf('图像文件名：%s\n', image_info.file_name);
fprintf('原始数据类型：%s\n', image_info.original_class);
fprintf('灰度图像尺寸：%d 行 x %d 列\n', M, N);
fprintf('灰度转换方式：%s\n', image_info.gray_method);
fprintf('sigma_list：%s\n', mat2str(sigma_list));
fprintf('默认展示 sigma：%.6g\n', default_sigma);
fprintf('核心频域链路：fft2 -> fftshift -> 频域相乘 -> ifftshift -> ifft2 -> real\n');

short_side = min(M, N);
if short_side < 128
    fprintf('提示：当前图像尺寸较小，sigma 可优先尝试 %.1f 到 %.1f。\n', ...
            max(2, short_side/32), max(3, short_side/8));
elseif short_side > 1024
    fprintf('提示：当前图像尺寸较大，可适当增大 sigma 或增大 mesh_downsample 以加快三维绘图。\n');
end
if any(sigma_list > short_side / 2)
    fprintf('提示：sigma_list 中存在较大的 sigma，低通滤波会保留较多细节，高通结果可能偏弱。\n');
end

%% ======================= 计算二维傅里叶频谱 =======================
% 二维离散傅里叶变换将图像从空间域 f(x,y) 变换到频域 F(u,v)。
% fftshift 将零频/低频分量移到频谱中心，便于构造以中心为原点的滤波器。
F = fft2(gray_image);
F_shift = fftshift(F);
original_log_spectrum = log(1 + abs(F_shift));
fprintf('原始频谱最大幅值：%.6e\n', max(abs(F_shift(:))));

%% ======================= 构造滤波器并执行频域高斯滤波 =======================
% 高斯低通滤波器：
% H_LP(u,v) = exp( -D(u,v)^2 / (2*sigma^2) )
% 高斯高通滤波器：
% H_HP(u,v) = 1 - H_LP(u,v)
%
% sigma 控制频域高斯滤波器的宽度：
% sigma 较小 -> 只允许更靠近中心的低频通过，低通结果更模糊，高通边缘更明显；
% sigma 较大 -> 允许更多频率通过，低通保留更多细节，高通只突出更高频部分。
results = struct([]);
for k = 1:numel(sigma_list)
    sigma = sigma_list(k);
    [H_LP, H_HP, U, V, D] = gaussian_frequency_filters(M, N, sigma);

    % 通过频谱与滤波器频率响应逐点相乘完成频域滤波：
    % G_LP_shift = F_shift .* H_LP;
    % G_HP_shift = F_shift .* H_HP;
    [lp_image, G_LP_shift, lp_imag_max] = apply_frequency_filter(F_shift, H_LP);
    [hp_image, G_HP_shift, hp_imag_max] = apply_frequency_filter(F_shift, H_HP);

    results(k).sigma = sigma;
    results(k).H_LP = H_LP;
    results(k).H_HP = H_HP;
    results(k).U = U;
    results(k).V = V;
    results(k).D = D;
    results(k).lp_image = lp_image;
    results(k).hp_image = hp_image;
    results(k).G_LP_shift = G_LP_shift;
    results(k).G_HP_shift = G_HP_shift;
    results(k).lp_mean = mean(lp_image(:));
    results(k).lp_std = std(lp_image(:));
    results(k).hp_mean = mean(hp_image(:));
    results(k).hp_std = std(hp_image(:));
    results(k).lp_imag_max = lp_imag_max;
    results(k).hp_imag_max = hp_imag_max;

    fprintf(['sigma = %.6g：低通均值 %.6f，低通标准差 %.6f；', ...
             '高通均值 %.6f，高通标准差 %.6f；最大虚部 %.3e / %.3e\n'], ...
             sigma, results(k).lp_mean, results(k).lp_std, ...
             results(k).hp_mean, results(k).hp_std, lp_imag_max, hp_imag_max);
end

% 理论上，灰度图像和对称实滤波器的逆变换结果应为实数。
% 由于浮点舍入误差，ifft2 后可能出现 1e-15 量级的极小虚部，
% apply_frequency_filter 中记录该虚部并使用 real() 取实部作为最终图像。
fprintf('说明：ifft2 后可能出现浮点舍入导致的极小虚部，代码已记录并使用 real() 取实部。\n');

[~, default_idx] = min(abs(sigma_list - default_sigma));
if abs(sigma_list(default_idx) - default_sigma) > eps(max(1, default_sigma))
    fprintf('提示：sigma_list 中没有 %.6g，默认展示改用最接近的 %.6g。\n', ...
            default_sigma, sigma_list(default_idx));
end
default_result = results(default_idx);

%% ======================= 保存单张结果图像 =======================
imwrite(normalize_for_display(gray_image, 'clip01'), fullfile(images_dir, 'original_gray.png'));
imwrite(normalize_for_display(original_log_spectrum, 'minmax'), fullfile(images_dir, 'original_spectrum_log.png'));

for k = 1:numel(results)
    sigma_tag = strrep(strrep(sprintf('%.6g', results(k).sigma), '.', 'p'), '-', 'm');
    imwrite(normalize_for_display(results(k).lp_image, 'clip01'), ...
            fullfile(images_dir, ['lowpass_result_sigma_' sigma_tag '.png']));
    imwrite(normalize_for_display(results(k).hp_image, 'minmax'), ...
            fullfile(images_dir, ['highpass_result_sigma_' sigma_tag '.png']));
    imwrite(normalize_for_display(results(k).H_LP, 'clip01'), ...
            fullfile(images_dir, ['gaussian_lowpass_filter_sigma_' sigma_tag '.png']));
    imwrite(normalize_for_display(results(k).H_HP, 'clip01'), ...
            fullfile(images_dir, ['gaussian_highpass_filter_sigma_' sigma_tag '.png']));
end

%% ======================= 绘制并保存综合对比图 =======================
plot_options.save_figures = save_figures;
plot_options.show_figures = show_figures;
plot_options.mesh_downsample = mesh_downsample;
plot_options.figures_dir = figures_dir;

show_filtering_results(original_display, gray_image, F_shift, default_result, results, plot_options);

%% ======================= 保存参数记录表 =======================
csv_path = fullfile(tables_dir, 'filter_parameter_record.csv');
try
    fid = fopen(csv_path, 'w', 'n', 'UTF-8');
catch
    fid = fopen(csv_path, 'w');
end
if fid < 0
    warning('无法写入参数记录表：%s', csv_path);
else
    fprintf(fid, '%s', char(65279));  % 写入 UTF-8 BOM，便于 Excel 正确识别中文
    fprintf(fid, '图像文件名,图像尺寸,sigma,低通结果均值,低通结果标准差,高通结果均值,高通结果标准差,备注\n');
    for k = 1:numel(results)
        fprintf(fid, '%s,%d x %d,%.6g,%.10f,%.10f,%.10f,%.10f,%s\n', ...
                image_info.file_name, M, N, results(k).sigma, ...
                results(k).lp_mean, results(k).lp_std, ...
                results(k).hp_mean, results(k).hp_std, ...
                '频域高斯滤波；统计量基于逆变换后的实部图像');
    end
    fclose(fid);
    fprintf('参数记录表已保存：%s\n', csv_path);
end

fprintf('单张结果图像保存目录：%s\n', images_dir);
fprintf('Figure 保存目录：%s\n', figures_dir);
fprintf('报告素材目录：%s\n', report_dir);
fprintf('全部流程完成。若需要重新实验，请修改参数区后再次运行 main_topicD_frequency_filter。\n');
