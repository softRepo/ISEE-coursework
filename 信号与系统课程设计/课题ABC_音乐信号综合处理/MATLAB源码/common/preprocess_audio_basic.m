function [audioOut, fsOut, stats, warningMessages] = preprocess_audio_basic(audioIn, fsIn, cfg)
%PREPROCESS_AUDIO_BASIC 单声道、去直流、重采样、归一化和淡入淡出
%   audioIn为原始或截取后的音频，fsIn为输入采样率。
%   cfg提供目标采样率、归一化峰值和淡入淡出时间等参数。

warningMessages = {};

if isempty(audioIn)
    error('输入音频为空，无法进行预处理。');
end

audio = double(audioIn);
audio(~isfinite(audio)) = 0;

channelCount = size(audio, 2);
if channelCount > 1
    audio = mean(audio, 2);
    channelProcess = sprintf('%d声道平均为单声道', channelCount);
else
    audio = audio(:);
    channelProcess = '原始单声道';
end

% 去除直流分量，避免后续频谱低频处出现无意义尖峰。
audio = audio - mean(audio);

fsOut = cfg.targetFs;
resampleMethod = '未重采样';
if fsIn ~= cfg.targetFs
    % 采样率不一致时，优先使用resample进行重采样。
    if exist('resample', 'file') == 2
        [p, q] = rat(cfg.targetFs / fsIn, 1e-10);
        audio = resample(audio, p, q);
        resampleMethod = sprintf('resample(%d/%d)', p, q);
    else
        oldTime = (0:numel(audio) - 1)' / fsIn;
        newLength = max(1, round(numel(audio) * cfg.targetFs / fsIn));
        newTime = (0:newLength - 1)' / cfg.targetFs;
        audio = interp1(oldTime, audio, newTime, 'linear', 'extrap');
        resampleMethod = '线性插值重采样';
        warningMessages{end + 1} = '未检测到resample函数，已使用线性插值完成重采样。';
    end
end

% 重采样后再次去直流，进一步降低数值误差带来的偏移。
audio = audio - mean(audio);

[audio, normalizeScale, peakBeforeNormalize] = normalize_audio_peak(audio, cfg.normalizedPeak);

fadeSamples = min(round(cfg.fadeInOutTime * fsOut), floor(numel(audio) / 2));
if fadeSamples > 1
    % 片段首尾加入短淡入淡出，降低播放时的突变感。
    fadeIn = linspace(0, 1, fadeSamples)';
    fadeOut = flipud(fadeIn);
    audio(1:fadeSamples) = audio(1:fadeSamples) .* fadeIn;
    audio(end - fadeSamples + 1:end) = audio(end - fadeSamples + 1:end) .* fadeOut;
end

audio(~isfinite(audio)) = 0;
audio = max(min(audio, 1), -1);
audioOut = audio(:);

stats = struct();
stats.channelProcess = channelProcess;
stats.resampleMethod = resampleMethod;
stats.peakBeforeNormalize = peakBeforeNormalize;
stats.normalizeScale = normalizeScale;
stats.peak = max(abs(audioOut));
stats.rms = sqrt(mean(audioOut .^ 2));
stats.fadeSamples = fadeSamples;
end
