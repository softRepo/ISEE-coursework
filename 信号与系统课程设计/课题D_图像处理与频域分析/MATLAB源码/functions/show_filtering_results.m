function fig_list = show_filtering_results(original_display, gray_image, F_shift, default_result, all_results, options)
%SHOW_FILTERING_RESULTS 生成并保存课题D要求的全部实验图形。
%
% 输入：
%   original_display - 原始图像显示数据，RGB 或灰度，范围 [0,1]。
%   gray_image       - 灰度归一化图像。
%   F_shift          - fftshift 后的原始图像频谱。
%   default_result   - 默认 sigma 对应的结果结构体。
%   all_results      - 所有 sigma 对应的结果结构体数组。
%   options          - 绘图选项，包含 show_figures、save_figures、
%                      mesh_downsample 和 figures_dir。
%
% 输出：
%   fig_list - Figure 句柄数组。
%
% 原理说明：
%   本函数负责实验结果可视化，滤波计算已经在主脚本中完成。三维滤波器图
%   仅在绘图时下采样，频域相乘计算始终使用完整尺寸滤波器。

if nargin < 6
    options = struct();
end
if ~isfield(options, 'show_figures')
    options.show_figures = true;
end
if ~isfield(options, 'save_figures')
    options.save_figures = true;
end
if ~isfield(options, 'mesh_downsample')
    options.mesh_downsample = 4;
end
if ~isfield(options, 'figures_dir')
    options.figures_dir = pwd;
end

visible_state = 'off';
if options.show_figures
    visible_state = 'on';
end

fig_list = {};
sigma = default_result.sigma;
sigma_tag = strrep(strrep(sprintf('%.6g', sigma), '.', 'p'), '-', 'm');

%% Figure 1：原始图像与频谱
fig1 = figure('Name', 'Figure 1 原始图像与频谱', 'Color', 'w', 'Visible', visible_state);
fig_list{end + 1} = fig1;

subplot(1, 3, 1);
show_original_image(original_display, '原始图像');

subplot(1, 3, 2);
show_gray_map(gray_image, '灰度归一化图像', '归一化灰度值', '水平像素', '垂直像素');

subplot(1, 3, 3);
show_gray_map(log(1 + abs(F_shift)), '原始图像对数幅度频谱', 'log(1+|F(u,v)|)', '水平频率索引', '垂直频率索引');

save_figure_safely(fig1, options.figures_dir, 'figure1_original_and_spectrum', options.save_figures);

%% Figure 2：某一组 sigma 下的滤波结果综合图
fig2 = figure('Name', 'Figure 2 默认sigma滤波结果综合图', 'Color', 'w', 'Visible', visible_state);
fig_list{end + 1} = fig2;

subplot(2, 3, 1);
show_gray_map(gray_image, '原始灰度图像', '归一化灰度值', '水平像素', '垂直像素');

subplot(2, 3, 2);
show_gray_map(default_result.H_LP, sprintf('高斯低通滤波器 sigma=%.6g', sigma), 'H_{LP}(u,v)', '水平频率 u', '垂直频率 v');

subplot(2, 3, 3);
show_gray_map(normalize_for_display(default_result.lp_image, 'clip01'), '低通滤波结果', '显示灰度值', '水平像素', '垂直像素');

subplot(2, 3, 4);
show_gray_map(default_result.H_HP, sprintf('高斯高通滤波器 sigma=%.6g', sigma), 'H_{HP}(u,v)', '水平频率 u', '垂直频率 v');

subplot(2, 3, 5);
show_gray_map(normalize_for_display(default_result.hp_image, 'minmax'), '高通滤波结果', '显示归一化幅值', '水平像素', '垂直像素');

subplot(2, 3, 6);
axis off;
text(0.02, 0.88, { ...
    sprintf('sigma = %.6g', sigma), ...
    '低通：保留低频轮廓，抑制高频细节，图像更平滑。', ...
    '高通：抑制低频背景，突出边缘、纹理和灰度突变。', ...
    '高通结果含正负值，显示时做了线性归一化。'}, ...
    'Units', 'normalized', 'FontSize', 10, 'Interpreter', 'none', ...
    'VerticalAlignment', 'top');

save_figure_safely(fig2, options.figures_dir, ['figure2_filtering_results_sigma_' sigma_tag], options.save_figures);

%% Figure 3：高斯低通和高斯高通滤波器三维频谱图
fig3 = figure('Name', 'Figure 3 高斯滤波器三维频谱图', 'Color', 'w', 'Visible', visible_state);
fig_list{end + 1} = fig3;

step = max(1, round(options.mesh_downsample));
row_idx = 1:step:size(default_result.H_LP, 1);
col_idx = 1:step:size(default_result.H_LP, 2);

subplot(1, 2, 1);
surf(default_result.U(row_idx, col_idx), default_result.V(row_idx, col_idx), default_result.H_LP(row_idx, col_idx));
title(sprintf('高斯低通三维频谱 sigma=%.6g', sigma), 'Interpreter', 'none');
xlabel('水平频率 u');
ylabel('垂直频率 v');
zlabel('幅度 H_{LP}(u,v)');
safe_colormap('jet');
safe_shading_interp();
colorbar;
view(45, 30);

subplot(1, 2, 2);
surf(default_result.U(row_idx, col_idx), default_result.V(row_idx, col_idx), default_result.H_HP(row_idx, col_idx));
title(sprintf('高斯高通三维频谱 sigma=%.6g', sigma), 'Interpreter', 'none');
xlabel('水平频率 u');
ylabel('垂直频率 v');
zlabel('幅度 H_{HP}(u,v)');
safe_colormap('jet');
safe_shading_interp();
colorbar;
view(45, 30);

save_figure_safely(fig3, options.figures_dir, ['figure3_filter_3d_spectrum_sigma_' sigma_tag], options.save_figures);

%% Figure 4：不同 sigma 参数下的低通结果对比
fig4 = figure('Name', 'Figure 4 不同sigma低通结果对比', 'Color', 'w', 'Visible', visible_state);
fig_list{end + 1} = fig4;
n = numel(all_results);
n_cols = min(3, n);
n_rows = ceil(n / n_cols);
for k = 1:n
    subplot(n_rows, n_cols, k);
    show_gray_map(normalize_for_display(all_results(k).lp_image, 'clip01'), ...
                  sprintf('低通结果 sigma=%.6g', all_results(k).sigma), ...
                  '显示灰度值', '水平像素', '垂直像素');
end
save_figure_safely(fig4, options.figures_dir, 'figure4_lowpass_sigma_comparison', options.save_figures);

%% Figure 5：不同 sigma 参数下的高通结果对比
fig5 = figure('Name', 'Figure 5 不同sigma高通结果对比', 'Color', 'w', 'Visible', visible_state);
fig_list{end + 1} = fig5;
for k = 1:n
    subplot(n_rows, n_cols, k);
    show_gray_map(normalize_for_display(all_results(k).hp_image, 'minmax'), ...
                  sprintf('高通结果 sigma=%.6g', all_results(k).sigma), ...
                  '显示归一化幅值', '水平像素', '垂直像素');
end
save_figure_safely(fig5, options.figures_dir, 'figure5_highpass_sigma_comparison', options.save_figures);

%% Figure 6：频域滤波前后频谱对比
fig6 = figure('Name', 'Figure 6 频域滤波前后频谱对比', 'Color', 'w', 'Visible', visible_state);
fig_list{end + 1} = fig6;

subplot(1, 3, 1);
show_gray_map(log(1 + abs(F_shift)), '原始频谱', 'log(1+|F|)', '水平频率索引', '垂直频率索引');

subplot(1, 3, 2);
show_gray_map(log(1 + abs(default_result.G_LP_shift)), ...
              sprintf('低通后频谱 sigma=%.6g', sigma), 'log(1+|G_{LP}|)', '水平频率索引', '垂直频率索引');

subplot(1, 3, 3);
show_gray_map(log(1 + abs(default_result.G_HP_shift)), ...
              sprintf('高通后频谱 sigma=%.6g', sigma), 'log(1+|G_{HP}|)', '水平频率索引', '垂直频率索引');

save_figure_safely(fig6, options.figures_dir, ['figure6_frequency_spectrum_comparison_sigma_' sigma_tag], options.save_figures);
end

function show_original_image(img, title_text)
%SHOW_ORIGINAL_IMAGE 显示原始图像，RGB 图像保持彩色显示。

if ndims(img) == 3 && size(img, 3) == 3
    image(img);
    axis image;
    set(gca, 'YDir', 'reverse');
    title(title_text, 'Interpreter', 'none');
    xlabel('水平像素');
    ylabel('垂直像素');
else
    show_gray_map(img, title_text, '归一化灰度值', '水平像素', '垂直像素');
end
end

function show_gray_map(data, title_text, colorbar_label, x_label, y_label)
%SHOW_GRAY_MAP 使用灰度色图显示二维矩阵，并添加中文坐标和色条。

imagesc(data);
axis image;
set(gca, 'YDir', 'reverse');
title(title_text, 'Interpreter', 'none');
xlabel(x_label);
ylabel(y_label);
safe_colormap('gray');
c = colorbar;
set_colorbar_label(c, colorbar_label);
end

function safe_colormap(map_name)
%SAFE_COLORMAP 兼容不同 MATLAB 版本的 colormap 调用。

try
    colormap(gca, map_name);
catch
    colormap(map_name);
end
end

function safe_shading_interp()
%SAFE_SHADING_INTERP 在支持时使用平滑着色。

try
    shading interp;
catch
    shading flat;
end
end

function set_colorbar_label(c, label_text)
%SET_COLORBAR_LABEL 兼容不同 MATLAB 版本的 colorbar 标签设置。

try
    ylabel(c, label_text, 'Interpreter', 'none');
catch
    try
        set(get(c, 'YLabel'), 'String', label_text, 'Interpreter', 'none');
    catch
        % 部分旧版本若设置失败，不影响主实验流程。
    end
end
end
