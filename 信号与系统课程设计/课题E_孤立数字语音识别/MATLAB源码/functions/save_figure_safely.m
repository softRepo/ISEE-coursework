function ok = save_figure_safely(fig, filePath, cfg)
%SAVE_FIGURE_SAFELY 按指定分辨率保存图像，并兼容不同 MATLAB 版本。

ok = false;
[folder, name, extName] = fileparts(filePath);
if isempty(extName)
    extName = '.png';
    filePath = fullfile(folder, [name extName]);
end
if ~exist(folder, 'dir')
    mkdir(folder);
end

try
    if exist('exportgraphics', 'file') || exist('exportgraphics', 'builtin')
        exportgraphics(fig, filePath, 'Resolution', cfg.figureDpi);
    else
        set(fig, 'PaperPositionMode', 'auto');
        print(fig, filePath, '-dpng', sprintf('-r%d', cfg.figureDpi));
    end

    if isfield(cfg, 'saveFigFile') && cfg.saveFigFile
        savefig(fig, fullfile(folder, [name '.fig']));
    end
    ok = true;
catch ME
    warning('Figure:SaveFailed', '保存图像失败：%s\n原因：%s', filePath, ME.message);
    try
        % 当常规保存方式失败时，使用 saveas 进行兼容保存。
        saveas(fig, filePath);
        ok = true;
    catch
        ok = false;
    end
end

if isfield(cfg, 'closeFiguresAfterSave') && cfg.closeFiguresAfterSave
    close(fig);
end

end
