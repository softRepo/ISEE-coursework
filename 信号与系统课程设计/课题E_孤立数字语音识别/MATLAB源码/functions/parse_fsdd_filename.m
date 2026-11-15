function [digitLabel, speakerName, recordIndex] = parse_fsdd_filename(fileName)
%PARSE_FSDD_FILENAME 解析 FSDD 文件名。
% 文件名格式为：{digitLabel}_{speakerName}_{index}.wav
% 输出数字标签、说话人名称和录音编号。

[~, baseName, extName] = fileparts(fileName);
if ~strcmpi(extName, '.wav')
    error('FSDD:InvalidExtension', '文件 %s 不是 wav 音频文件。', fileName);
end

tokens = regexp([baseName extName], '^(\d+)_(.+)_(\d+)\.wav$', 'tokens', 'once');
if isempty(tokens)
    error('FSDD:InvalidFileName', ...
        '文件名 %s 不符合 {digit}_{speaker}_{index}.wav 格式。', fileName);
end

digitLabel = str2double(tokens{1});
speakerName = tokens{2};
recordIndex = str2double(tokens{3});

if isnan(digitLabel) || isnan(recordIndex)
    error('FSDD:InvalidNumber', '文件名 %s 中的数字标签或编号无法解析。', fileName);
end

end
