function [x, fs, info] = load_audio_segment(filePath, targetFs)
%LOAD_AUDIO_SEGMENT 读取实验用wav片段并统一为单声道double列向量
%   filePath为片段路径，targetFs为可选目标采样率。
%   函数完成声道合并、去直流和必要的重采样处理。

if nargin < 2
    targetFs = [];
end
if ~exist(filePath, 'file')
    error('未找到音频片段：%s', filePath);
end

info = audioinfo(filePath);
[x, fs] = audioread(filePath);
x = double(x);
if size(x, 2) > 1
    % 多声道音频按通道平均转换为单声道。
    x = mean(x, 2);
else
    x = x(:);
end
x = x - mean(x);

if ~isempty(targetFs) && fs ~= targetFs
    if exist('resample', 'file') == 2
        [p, q] = rat(targetFs / fs, 1e-10);
        x = resample(x, p, q);
    else
        oldT = (0:numel(x)-1)' / fs;
        newN = round(numel(x) * targetFs / fs);
        newT = (0:newN-1)' / targetFs;
        x = interp1(oldT, x, newT, 'linear', 'extrap');
    end
    fs = targetFs;
end
x = max(min(x, 1), -1);
end
