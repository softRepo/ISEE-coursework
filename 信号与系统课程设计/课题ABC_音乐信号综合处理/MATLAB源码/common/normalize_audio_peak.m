function [audioOut, scaleFactor, peakValue] = normalize_audio_peak(audioIn, targetPeak)
%NORMALIZE_AUDIO_PEAK 将音频峰值归一化到指定目标峰值
%   targetPeak用于设置归一化后的最大绝对幅值。
%   当输入为静音或近似静音时，保持原信号不变。

if nargin < 2 || isempty(targetPeak)
    targetPeak = 0.98;
end

audio = double(audioIn);
audio(~isfinite(audio)) = 0;
peakValue = max(abs(audio(:)));

if peakValue <= eps
    audioOut = audio;
    scaleFactor = 1;
else
    scaleFactor = targetPeak / peakValue;
    audioOut = audio * scaleFactor;
end
end
