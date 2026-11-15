function saved_paths = save_figure_safely(fig_handle, output_dir, base_name, save_figures)
%SAVE_FIGURE_SAFELY 将 Figure 同时保存为 .png 和 .fig。
%
% 输入：
%   fig_handle   - Figure 句柄。
%   output_dir   - 保存目录。
%   base_name    - 不含扩展名的文件名。
%   save_figures - true/false，控制是否实际保存。
%
% 输出：
%   saved_paths - 结构体，包含 png 和 fig 两个输出路径。
%
% 原理说明：
%   新版本 MATLAB 优先使用 exportgraphics 输出 PNG，以获得较清晰的
%   位图；若 exportgraphics 不可用或失败，则改用 saveas。
%   .fig 文件优先使用 savefig，旧版本环境则改用 saveas。

saved_paths = struct('png', '', 'fig', '');

if nargin < 4
    save_figures = true;
end
if ~save_figures
    return;
end
if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

png_path = fullfile(output_dir, [base_name, '.png']);
fig_path = fullfile(output_dir, [base_name, '.fig']);
saved_paths.png = png_path;
saved_paths.fig = fig_path;

try
    if exist('savefig', 'file') == 2 || exist('savefig', 'builtin') == 5
        savefig(fig_handle, fig_path);
    else
        saveas(fig_handle, fig_path);
    end
catch ME
    warning('保存 FIG 文件失败：%s\n原因：%s', fig_path, ME.message);
end

try
    if exist('exportgraphics', 'file') == 2 || exist('exportgraphics', 'builtin') == 5
        exportgraphics(fig_handle, png_path, 'Resolution', 200);
    else
        saveas(fig_handle, png_path);
    end
catch
    try
        saveas(fig_handle, png_path);
    catch ME
        warning('保存 PNG 文件失败：%s\n原因：%s', png_path, ME.message);
    end
end
end
