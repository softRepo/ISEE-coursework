function [frames, frameTimeSec] = frame_signal(x, frameLength, frameShift, fs, padMode)
%FRAME_SIGNAL 将一维语音信号按固定帧长和帧移分帧。
% 输出矩阵 frames 的每一行表示一帧，列方向表示帧内采样点。

if nargin < 5 || isempty(padMode)
    padMode = 'discard';
end
if nargin < 4 || isempty(fs)
    fs = 1;
end

x = double(x(:));
frameLength = max(1, round(frameLength));
frameShift = max(1, round(frameShift));

if isempty(x)
    frames = zeros(0, frameLength);
    frameTimeSec = zeros(0, 1);
    return;
end

signalLength = numel(x);
if strcmpi(padMode, 'zeropad')
    if signalLength <= frameLength
        numFrames = 1;
    else
        numFrames = ceil((signalLength - frameLength) / frameShift) + 1;
    end
    paddedLength = (numFrames - 1) * frameShift + frameLength;
    if paddedLength > signalLength
        x = [x; zeros(paddedLength - signalLength, 1)];
    end
else
    if signalLength < frameLength
        frames = zeros(0, frameLength);
        frameTimeSec = zeros(0, 1);
        return;
    end
    numFrames = floor((signalLength - frameLength) / frameShift) + 1;
end

idx = (1:frameLength)' + (0:numFrames-1) * frameShift;
frames = x(idx).';
frameCenter = (0:numFrames-1)' * frameShift + (frameLength + 1) / 2;
frameTimeSec = (frameCenter - 1) / fs;

end
