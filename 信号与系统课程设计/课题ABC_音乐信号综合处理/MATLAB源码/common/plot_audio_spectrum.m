function [freq, mag, magDb] = plot_audio_spectrum(audioData, fs)
%PLOT_AUDIO_SPECTRUM 计算单边FFT幅度谱和dB谱
%   函数返回频率轴、线性幅度谱和dB谱，可供检查图或后续分析使用。

audio = double(audioData(:));
audio(~isfinite(audio)) = 0;
audio = audio - mean(audio);

n = numel(audio);
if n == 0
    freq = [];
    mag = [];
    magDb = [];
    return;
end

nfft = 2 ^ nextpow2(n);
spectrum = fft(audio, nfft);
halfCount = floor(nfft / 2) + 1;

mag = abs(spectrum(1:halfCount)) / n;
if halfCount > 2
    mag(2:end - 1) = 2 * mag(2:end - 1);
end
freq = (0:halfCount - 1)' * fs / nfft;
magDb = 20 * log10(mag + eps);
end
