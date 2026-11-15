function write_latex_table(headers, rows, filePath)
%WRITE_LATEX_TABLE 写出表格排版文本
%   headers为表头，rows为表格内容，filePath为输出文件路径。
%   函数用于保存核心指标表，便于后续整理实验材料。

targetDir = fileparts(filePath);
if ~exist(targetDir, 'dir')
    mkdir(targetDir);
end
[fid, msg] = fopen(filePath, 'w', 'n', 'UTF-8');
if fid < 0
    error('无法写出LaTeX表格：%s。系统提示：%s', filePath, msg);
end
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

colSpec = ['l', repmat('c', 1, numel(headers)-1)];
fprintf(fid, '\\begin{tabular}{%s}\n', colSpec);
fprintf(fid, '\\toprule\n');
fprintf(fid, '%s \\\\\n', strjoin(cellfun(@latex_escape, headers, 'UniformOutput', false), ' & '));
fprintf(fid, '\\midrule\n');
for i = 1:size(rows, 1)
    rowText = cell(1, size(rows, 2));
    for j = 1:size(rows, 2)
        rowText{j} = latex_escape(rows{i, j});
    end
    fprintf(fid, '%s \\\\\n', strjoin(rowText, ' & '));
end
fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');
end

function out = latex_escape(value)
%LATEX_ESCAPE 转义表格文本中的特殊字符
if isnumeric(value)
    if isscalar(value)
        out = sprintf('%.4g', value);
    else
        out = mat2str(value, 4);
    end
else
    out = char(string(value));
end
out = strrep(out, '\', '\textbackslash{}');
out = strrep(out, '_', '\_');
out = strrep(out, '%', '\%');
out = strrep(out, '&', '\&');
out = strrep(out, '#', '\#');
end
