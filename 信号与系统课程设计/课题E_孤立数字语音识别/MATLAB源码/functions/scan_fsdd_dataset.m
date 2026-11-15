function dataTbl = scan_fsdd_dataset(recordingsDir)
%SCAN_FSDD_DATASET 扫描 FSDD recordings 目录并解析样本标签信息。
% 输出表格包含样本路径、数字标签、说话人名称和录音编号。

if ~exist(recordingsDir, 'dir')
    error('FSDD:MissingRecordingsDir', '找不到 recordings 目录：%s', recordingsDir);
end

wavFiles = dir(fullfile(recordingsDir, '*.wav'));
if isempty(wavFiles)
    error('FSDD:NoWavFiles', 'recordings 目录中没有找到 wav 文件：%s', recordingsDir);
end

n = numel(wavFiles);
fileName = cell(n, 1);
filePath = cell(n, 1);
speaker = cell(n, 1);
digit = zeros(n, 1);
recordIndex = zeros(n, 1);

for k = 1:n
    fileName{k} = wavFiles(k).name;
    filePath{k} = fullfile(wavFiles(k).folder, wavFiles(k).name);
    [digit(k), speaker{k}, recordIndex(k)] = parse_fsdd_filename(wavFiles(k).name);
end

dataTbl = table(fileName, filePath, digit, speaker, recordIndex, ...
    'VariableNames', {'FileName', 'FilePath', 'Digit', 'Speaker', 'Index'});
dataTbl = sortrows(dataTbl, {'Digit', 'Speaker', 'Index'});

end
