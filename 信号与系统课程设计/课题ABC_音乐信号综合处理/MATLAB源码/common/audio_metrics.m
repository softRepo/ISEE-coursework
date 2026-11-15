function m = audio_metrics(x, fs, reference)
%AUDIO_METRICS 计算音频基础指标和相对误差指标
%   x为待评价音频信号，fs为采样率，reference为可选参考信号。
%   当提供参考信号时，函数计算MSE、RMSE、相关系数和SNR。

if nargin < 3
    reference = [];
end
x = double(x(:));
x(~isfinite(x)) = 0;

m = struct();
m.duration = numel(x) / fs;
m.peak = max(abs(x));
m.rms = sqrt(mean(x .^ 2));
m.energy = sum(x .^ 2);
m.dc = mean(x);

if ~isempty(reference)
    % 与参考信号对齐长度后计算重构误差和相似度指标。
    reference = double(reference(:));
    n = min(numel(reference), numel(x));
    ref = reference(1:n);
    y = x(1:n);
    err = ref - y;
    m.mse = mean(err .^ 2);
    m.rmse = sqrt(m.mse);
    if std(ref) <= eps || std(y) <= eps
        m.corr = 0;
    else
        c = corrcoef(ref, y);
        m.corr = c(1, 2);
    end
    m.snr = 10 * log10(sum(ref .^ 2) / max(sum(err .^ 2), eps));
else
    m.mse = NaN;
    m.rmse = NaN;
    m.corr = NaN;
    m.snr = NaN;
end
end
