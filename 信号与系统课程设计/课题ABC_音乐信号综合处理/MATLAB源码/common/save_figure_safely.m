function save_figure_safely(figHandle, filePath)
%SAVE_FIGURE_SAFELY 保存图像文件并兼容不同MATLAB版本
%   优先使用exportgraphics；不可用时使用saveas保存图像。
%   保存前自动创建目标目录。

targetDir = fileparts(filePath);
if ~exist(targetDir, 'dir')
    mkdir(targetDir);
end

try
    exportgraphics(figHandle, filePath, 'Resolution', 200);
catch
    try
        saveas(figHandle, filePath);
    catch ME
        error('保存图像失败：%s。原始错误：%s', filePath, ME.message);
    end
end
end
