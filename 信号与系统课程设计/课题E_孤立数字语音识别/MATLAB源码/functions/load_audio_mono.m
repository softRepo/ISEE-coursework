function [x, fs] = load_audio_mono(filePath)
%LOAD_AUDIO_MONO 读取 wav 音频，并转换为单声道 double 列向量。

if ~exist(filePath, 'file')
    error('Audio:MissingFile', '找不到音频文件：%s', filePath);
end

[x, fs] = audioread(filePath);
x = double(x);

if size(x, 2) > 1
    x = mean(x, 2);
end

x = x(:);
x(~isfinite(x)) = 0;

end
