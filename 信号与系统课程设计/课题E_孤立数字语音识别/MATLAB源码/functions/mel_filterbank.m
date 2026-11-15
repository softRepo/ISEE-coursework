function [filterBank, centerFreqHz] = mel_filterbank(numFilters, nfft, fs, lowHz, highHz)
%MEL_FILTERBANK 构造 Mel 三角滤波器组。
% 每一行对应一个三角滤波器，每一列对应一个 FFT 正频率频点。

if nargin < 4 || isempty(lowHz)
    lowHz = 0;
end
if nargin < 5 || isempty(highHz)
    highHz = fs / 2;
end

numFreqBins = floor(nfft / 2) + 1;
lowHz = max(0, lowHz);
highHz = min(fs / 2, highHz);

melLow = hz2mel_local(lowHz);
melHigh = hz2mel_local(highHz);
melPoints = linspace(melLow, melHigh, numFilters + 2);
hzPoints = mel2hz_local(melPoints);
binPoints = floor((nfft + 1) * hzPoints / fs);
binPoints = min(max(binPoints, 0), numFreqBins - 1);

filterBank = zeros(numFilters, numFreqBins);
for m = 1:numFilters
    left = binPoints(m);
    center = binPoints(m + 1);
    right = binPoints(m + 2);

    if center <= left
        center = left + 1;
    end
    if right <= center
        right = center + 1;
    end
    right = min(right, numFreqBins - 1);

    for k = left:center
        filterBank(m, k + 1) = (k - left) / max(center - left, eps);
    end
    for k = center:right
        filterBank(m, k + 1) = (right - k) / max(right - center, eps);
    end
end

centerFreqHz = hzPoints(2:end-1);

end
