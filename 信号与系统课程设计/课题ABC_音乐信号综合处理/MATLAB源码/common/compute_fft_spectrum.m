function spec = compute_fft_spectrum(x, fs)
%COMPUTE_FFT_SPECTRUM 计算单边FFT幅度谱、dB谱和功率谱
%   输入信号先转换为列向量并去除直流分量，再进行FFT分析。
%   输出结构体包含频率轴、线性幅度谱、dB谱、功率谱和FFT点数。

x = double(x(:));
x(~isfinite(x)) = 0;
x = x - mean(x);
n = numel(x);
nfft = 2 ^ nextpow2(max(n, 2));
X = fft(x, nfft);
halfN = floor(nfft / 2) + 1;
freq = (0:halfN-1)' * fs / nfft;
amp = abs(X(1:halfN)) / n;
if halfN > 2
    amp(2:end-1) = 2 * amp(2:end-1);
end
power = amp .^ 2;

spec = struct();
spec.freq = freq;
spec.amp = amp;
spec.db = 20 * log10(amp + eps);
spec.power = power;
spec.nfft = nfft;
end
