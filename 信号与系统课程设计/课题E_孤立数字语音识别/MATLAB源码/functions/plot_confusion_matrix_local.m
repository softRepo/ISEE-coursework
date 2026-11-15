function fig = plot_confusion_matrix_local(confMat, digitLabels, filePath, cfg)
%PLOT_CONFUSION_MATRIX_LOCAL 绘制十分类识别混淆矩阵并保存图像。

fig = figure('Visible', cfg.figureVisible, 'Color', 'w', 'Name', '十分类混淆矩阵');
imagesc(confMat);
axis image;
colormap(parula);
colorbar;

numLabels = numel(digitLabels);
set(gca, 'XTick', 1:numLabels, 'XTickLabel', digitLabels, ...
    'YTick', 1:numLabels, 'YTickLabel', digitLabels, 'FontName', 'Microsoft YaHei');
xlabel('预测数字');
ylabel('真实数字');
title('孤立数字语音识别混淆矩阵');

maxValue = max(confMat(:));
if maxValue < 1
    maxValue = 1;
end
for r = 1:size(confMat, 1)
    for c = 1:size(confMat, 2)
        if confMat(r, c) > maxValue * 0.55
            textColor = 'w';
        else
            textColor = 'k';
        end
        text(c, r, sprintf('%d', confMat(r, c)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Color', textColor, 'FontWeight', 'bold');
    end
end

save_figure_safely(fig, filePath, cfg);

end
