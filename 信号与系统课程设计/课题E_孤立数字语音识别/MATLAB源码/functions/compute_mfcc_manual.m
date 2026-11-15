function [features, info] = compute_mfcc_manual(x, fs, cfg)
%COMPUTE_MFCC_MANUAL 手工提取 MFCC 特征，不调用 MATLAB mfcc 函数。
% 处理流程：分帧 -> 汉明窗 -> FFT -> 功率谱 -> Mel 滤波器组 ->
% 对数能量 -> DCT-II -> 倒谱系数选取 -> 可选差分特征。

x = double(x(:));
x(~isfinite(x)) = 0;
if isempty(x)
    features = zeros(0, cfg.numMfcc);
    info = struct('warning', '输入语音为空。');
    return;
end

frameLength = round(cfg.frameLengthMs / 1000 * fs);
frameShift = round(cfg.frameShiftMs / 1000 * fs);
[frames, frameTimeSec] = frame_signal(x, frameLength, frameShift, fs, 'zeropad');

if isempty(frames)
    frames = x(:).';
    if size(frames, 2) < frameLength
        frames = [frames zeros(1, frameLength - size(frames, 2))];
    end
    frameTimeSec = 0;
end

win = local_hamming(frameLength).';
framesWin = frames .* repmat(win, size(frames, 1), 1);

nfft = max(cfg.nfft, 2 ^ nextpow2(frameLength));
fftFrames = fft(framesWin, nfft, 2);
powSpec = (abs(fftFrames(:, 1:floor(nfft/2)+1)) .^ 2) / nfft;

[fb, centerFreqHz] = mel_filterbank(cfg.numMelFilters, nfft, fs, cfg.melLowHz, cfg.melHighHz);
melEnergy = powSpec * fb.';
melEnergy = max(melEnergy, eps);
logMelEnergy = log(melEnergy);

% 使用 DCT-II 公式手工构造变换矩阵，将对数 Mel 能量变换为倒谱系数。
numNeeded = cfg.numMfcc + double(~cfg.useC0);
dctMatrix = dct2_manual(numNeeded, cfg.numMelFilters);
mfccAll = logMelEnergy * dctMatrix.';
if cfg.useC0
    mfcc = mfccAll(:, 1:cfg.numMfcc);
else
    mfcc = mfccAll(:, 2:cfg.numMfcc+1);
end

features = mfcc;
if cfg.useDelta
    delta = compute_delta_features(mfcc, cfg.deltaWindow);
    features = [features delta];
    if cfg.useDeltaDelta
        deltaDelta = compute_delta_features(delta, cfg.deltaWindow);
        features = [features deltaDelta];
    end
end

if cfg.useCepstralMeanVarianceNorm && ~isempty(features)
    mu = mean(features, 1);
    sigma = std(features, 0, 1);
    sigma(sigma < 1e-8) = 1;
    features = (features - repmat(mu, size(features, 1), 1)) ./ repmat(sigma, size(features, 1), 1);
end

features(~isfinite(features)) = 0;

info = struct();
info.frameLength = frameLength;
info.frameShift = frameShift;
info.frameTimeSec = frameTimeSec;
info.nfft = nfft;
info.melFilterbank = fb;
info.centerFreqHz = centerFreqHz;
info.logMelEnergy = logMelEnergy;
info.mfccStatic = mfcc;
info.featureDim = size(features, 2);
info.warning = '';

end

function w = local_hamming(n)
if n <= 1
    w = 1;
else
    k = (0:n-1)';
    w = 0.54 - 0.46 * cos(2 * pi * k / (n - 1));
end
end
