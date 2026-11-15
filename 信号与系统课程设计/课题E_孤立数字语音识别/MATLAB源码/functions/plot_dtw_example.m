function plot_dtw_example(testFeatures, templateFeatures, dtwInfo, outputDir, prefix, cfg)
%PLOT_DTW_EXAMPLE 绘制并保存 DTW 局部距离矩阵、累积距离矩阵和最优路径图。

if nargin < 5 || isempty(prefix)
    prefix = 'dtw_example';
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fig1 = figure('Visible', cfg.figureVisible, 'Color', 'w', 'Name', 'DTW局部距离矩阵');
imagesc(dtwInfo.localDistance);
axis xy;
colorbar;
xlabel('模板语音帧序号');
ylabel('测试语音帧序号');
title('DTW 局部距离矩阵 d(i,j)');
set(gca, 'FontName', 'Microsoft YaHei');
save_figure_safely(fig1, fullfile(outputDir, [prefix '_local_distance.png']), cfg);

fig2 = figure('Visible', cfg.figureVisible, 'Color', 'w', 'Name', 'DTW累积距离与路径');
imagesc(dtwInfo.accumulatedDistance);
axis xy;
hold on;
if isfield(dtwInfo, 'path') && ~isempty(dtwInfo.path)
    plot(dtwInfo.path(:, 2), dtwInfo.path(:, 1), 'w-', 'LineWidth', 2);
    plot(dtwInfo.path(:, 2), dtwInfo.path(:, 1), 'k--', 'LineWidth', 0.8);
end
hold off;
colorbar;
xlabel('模板语音帧序号');
ylabel('测试语音帧序号');
title('DTW 累积距离矩阵 D(i,j) 与最优路径');
legend({'最优路径'}, 'Location', 'southeast');
set(gca, 'FontName', 'Microsoft YaHei');
save_figure_safely(fig2, fullfile(outputDir, [prefix '_accumulated_path.png']), cfg);

fig3 = figure('Visible', cfg.figureVisible, 'Color', 'w', 'Name', 'DTW特征长度对比');
plot(testFeatures(:, 1), 'LineWidth', 1.2);
hold on;
plot(templateFeatures(:, 1), 'LineWidth', 1.2);
hold off;
grid on;
xlabel('帧序号');
ylabel('第 1 维 MFCC 标准化系数');
title('测试样本与最佳模板的 MFCC 序列对比');
legend({'测试样本', '最佳模板'}, 'Location', 'best');
set(gca, 'FontName', 'Microsoft YaHei');
save_figure_safely(fig3, fullfile(outputDir, [prefix '_mfcc_sequence_compare.png']), cfg);

end
