function stftData = compute_stft_manual(x, fs, winMs, hopRatio)
%COMPUTE_STFT_MANUAL 手工分帧、加窗和FFT计算STFT语谱图
%   winMs为窗长，hopRatio为帧移比例。
%   该函数用于在不同MATLAB版本下稳定获得时频分析结果。

if nargin < 4 || isempty(hopRatio)
    hopRatio = 0.5;
end
x = double(x(:));
x(~isfinite(x)) = 0;

winLength = max(64, round(winMs / 1000 * fs));
winLength = min(winLength, numel(x));
hop = max(1, round(winLength * hopRatio));
nfft = max(512, 2 ^ nextpow2(winLength));

n = (0:winLength-1)';
win = 0.54 - 0.46 * cos(2*pi*n/max(winLength-1, 1));
if numel(x) < winLength
    x = [x; zeros(winLength-numel(x), 1)];
end
frameCount = 1 + floor((numel(x) - winLength) / hop);
halfN = floor(nfft / 2) + 1;
S = zeros(halfN, frameCount);

for iFrame = 1:frameCount
    % 对每一帧加窗后进行FFT，保留单边频谱。
    idx = (iFrame-1) * hop + (1:winLength);
    frame = x(idx) .* win;
    X = fft(frame, nfft);
    S(:, iFrame) = X(1:halfN);
end

magDb = 20 * log10(abs(S) + eps);
magDb = max(magDb, max(magDb(:)) - 100);

stftData = struct();
stftData.S = S;
stftData.magDb = magDb;
stftData.freq = (0:halfN-1)' * fs / nfft;
stftData.time = ((0:frameCount-1) * hop + winLength/2) / fs;
stftData.winLength = winLength;
stftData.hop = hop;
stftData.nfft = nfft;
end
