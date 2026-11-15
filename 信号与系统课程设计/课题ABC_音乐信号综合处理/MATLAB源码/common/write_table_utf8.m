function write_table_utf8(T, filePath)
%WRITE_TABLE_UTF8 以UTF-8编码写出表格
%   T为MATLAB表格，filePath为csv输出路径。
%   当当前MATLAB版本不支持Encoding参数时，使用兼容方式写出。

targetDir = fileparts(filePath);
if ~exist(targetDir, 'dir')
    mkdir(targetDir);
end
try
    writetable(T, filePath, 'Encoding', 'UTF-8');
catch
    writetable(T, filePath);
end
end
