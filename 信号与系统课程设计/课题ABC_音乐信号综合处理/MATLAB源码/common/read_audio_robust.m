function [audioData, fs, meta] = read_audio_robust(filePath)
%READ_AUDIO_ROBUST 稳健读取音频并计算基础信息
%   filePath为待读取音频文件路径。
%   对格式不支持、文件损坏或读取失败的情况给出中文错误提示。

if ~exist(filePath, 'file')
    error('音频文件不存在：%s', filePath);
end

try
    info = audioinfo(filePath);
catch ME
    error(['无法读取音频文件信息：%s', newline, ...
        '可能是MATLAB不支持该格式或文件已损坏。原始错误：%s'], filePath, ME.message);
end

try
    [audioData, fs] = audioread(filePath);
catch ME
    error(['无法读取音频文件：%s', newline, ...
        '如果这是MATLAB不支持的格式，请先转换为wav后再运行。原始错误：%s'], ...
        filePath, ME.message);
end

audioData = double(audioData);
audioData(~isfinite(audioData)) = 0;

[~, fileName, ext] = fileparts(filePath);
meta = struct();
meta.fileName = [fileName, ext];
meta.filePath = filePath;
meta.sampleRate = fs;
meta.channels = size(audioData, 2);
meta.duration = size(audioData, 1) / fs;
meta.dataType = class(audioData);
meta.audioInfo = info;

if isempty(audioData)
    meta.peak = 0;
    meta.rms = 0;
else
    meta.peak = max(abs(audioData(:)));
    meta.rms = sqrt(mean(audioData(:) .^ 2));
end
end
