function endpoint = endpoint_detect_energy_zcr(x, fs, cfg)
%ENDPOINT_DETECT_ENERGY_ZCR 基于短时能量和过零率的自适应端点检测。
% 短时能量用于定位主要语音段，过零率用于辅助保留低能量清辅音片段。

x = double(x(:));
frameLength = round(cfg.frameLengthMs / 1000 * fs);
frameShift = round(cfg.frameShiftMs / 1000 * fs);
[frames, frameTimeSec] = frame_signal(x, frameLength, frameShift, fs, 'zeropad');

if isempty(frames)
    endpoint = make_failed_endpoint(frameLength, frameShift);
    endpoint.warning = '音频过短，无法分帧。';
    return;
end

win = local_hamming(frameLength).';
windowedFrames = frames .* repmat(win, size(frames, 1), 1);
energy = sum(windowedFrames .^ 2, 2);

signFrames = frames;
signFrames(signFrames >= 0) = 1;
signFrames(signFrames < 0) = -1;
zcr = sum(abs(diff(signFrames, 1, 2)), 2) / (2 * frameLength);

energyMax = max(energy);
zcrMax = max(zcr);
if energyMax < eps || numel(x) < frameLength
    endpoint = make_failed_endpoint(frameLength, frameShift);
    endpoint.energy = energy;
    endpoint.zcr = zcr;
    endpoint.frameTimeSec = frameTimeSec;
    endpoint.warning = '音频能量过低或长度过短，端点检测回退。';
    return;
end

energyNorm = energy / (energyMax + eps);
zcrNorm = zcr / (zcrMax + eps);

edgeCount = max(2, min(5, floor(numel(energyNorm) / 4)));
noiseEnergy = median([energyNorm(1:edgeCount); energyNorm(end-edgeCount+1:end)]);
energyThreshold = noiseEnergy + cfg.endpointEnergyRatio * (1 - noiseEnergy);
energyThreshold = min(max(energyThreshold, 0.025), 0.35);

zcrMedian = median(zcrNorm);
zcrMad = median(abs(zcrNorm - zcrMedian)) + eps;
zcrThreshold = zcrMedian + cfg.endpointZcrRatio * zcrMad;
zcrThreshold = min(max(zcrThreshold, 0.03), 0.55);

energyActive = energyNorm >= energyThreshold;
zcrActive = (energyNorm >= 0.45 * energyThreshold) & (zcrNorm >= zcrThreshold);
active = energyActive | zcrActive;

smoothFrames = max(1, round(cfg.endpointSmoothFrames));
active = conv(double(active), ones(smoothFrames, 1), 'same') > 0;

activeIdx = find(active);
endpoint = struct();
endpoint.energy = energy;
endpoint.zcr = zcr;
endpoint.energyNorm = energyNorm;
endpoint.zcrNorm = zcrNorm;
endpoint.frameTimeSec = frameTimeSec;
endpoint.energyThreshold = energyThreshold;
endpoint.zcrThreshold = zcrThreshold;
endpoint.frameLength = frameLength;
endpoint.frameShift = frameShift;

if isempty(activeIdx)
    endpoint.success = false;
    endpoint.startFrame = 1;
    endpoint.endFrame = size(frames, 1);
    endpoint.startSample = 1;
    endpoint.endSample = numel(x);
    endpoint.warning = '没有检测到稳定语音端点，回退到完整音频。';
    return;
end

guard = max(0, round(cfg.endpointGuardFrames));
startFrame = max(1, activeIdx(1) - guard);
endFrame = min(size(frames, 1), activeIdx(end) + guard);
startSample = (startFrame - 1) * frameShift + 1;
endSample = min(numel(x), (endFrame - 1) * frameShift + frameLength);

minSpeechSamples = round(cfg.minSpeechDurationSec * fs);
if endSample - startSample + 1 < minSpeechSamples
    endpoint.success = false;
    endpoint.startFrame = 1;
    endpoint.endFrame = size(frames, 1);
    endpoint.startSample = 1;
    endpoint.endSample = numel(x);
    endpoint.warning = '检测到的有效语音过短，回退到完整音频。';
else
    endpoint.success = true;
    endpoint.startFrame = startFrame;
    endpoint.endFrame = endFrame;
    endpoint.startSample = startSample;
    endpoint.endSample = endSample;
    endpoint.warning = '';
end

end

function endpoint = make_failed_endpoint(frameLength, frameShift)
endpoint = struct();
endpoint.success = false;
endpoint.startFrame = 1;
endpoint.endFrame = 1;
endpoint.startSample = 1;
endpoint.endSample = 1;
endpoint.energy = [];
endpoint.zcr = [];
endpoint.energyNorm = [];
endpoint.zcrNorm = [];
endpoint.frameTimeSec = [];
endpoint.energyThreshold = NaN;
endpoint.zcrThreshold = NaN;
endpoint.frameLength = frameLength;
endpoint.frameShift = frameShift;
endpoint.warning = '';
end

function w = local_hamming(n)
if n <= 1
    w = 1;
else
    k = (0:n-1)';
    w = 0.54 - 0.46 * cos(2 * pi * k / (n - 1));
end
end
