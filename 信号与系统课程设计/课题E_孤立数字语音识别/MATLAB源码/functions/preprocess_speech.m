function [speech, info] = preprocess_speech(x, fs, cfg)
%PREPROCESS_SPEECH 对语音信号进行预处理并提取有效语音片段。
% 处理内容包括单声道转换、去直流、归一化、重采样、预加重和端点检测。

info = struct();
info.originalFs = fs;
info.targetFs = cfg.targetFs;
info.warning = '';

x = double(x);
if size(x, 2) > 1
    x = mean(x, 2);
end
x = x(:);
x(~isfinite(x)) = 0;
info.originalSignal = x;

if isempty(x)
    speech = zeros(0, 1);
    info.warning = '输入音频为空。';
    return;
end

% 去直流分量，避免低频直流偏置影响短时能量和 MFCC。
x = x - mean(x);
x = normalize_audio(x);

% FSDD 默认采样率为 8 kHz；必要时使用线性插值统一到目标采样率。
if fs ~= cfg.targetFs
    oldTime = (0:numel(x)-1)' / fs;
    newTime = (0:round(oldTime(end) * cfg.targetFs))' / cfg.targetFs;
    x = interp1(oldTime, x, newTime, 'linear', 'extrap');
    fs = cfg.targetFs;
end
info.normalizedSignal = x;

if cfg.enablePreEmphasis && numel(x) >= 2
    % 预加重公式：y[n] = x[n] - a x[n-1]，用于提升高频语音细节。
    xPre = [x(1); x(2:end) - cfg.preEmphasisAlpha * x(1:end-1)];
else
    xPre = x;
end
xPre = normalize_audio(xPre);
info.preEmphasizedSignal = xPre;

endpoint = endpoint_detect_energy_zcr(xPre, fs, cfg);
info.endpoint = endpoint;

if endpoint.success
    speech = xPre(endpoint.startSample:endpoint.endSample);
else
    speech = xPre;
    info.warning = endpoint.warning;
end

speech = normalize_audio(speech);
info.outputLength = numel(speech);
info.outputDurationSec = numel(speech) / fs;

end
