function delta = compute_delta_features(features, deltaWindow)
%COMPUTE_DELTA_FEATURES 计算一阶或二阶动态差分特征。
% 差分特征反映语音谱包络随时间的变化，用于补充 MFCC 静态特征。

if nargin < 2 || isempty(deltaWindow)
    deltaWindow = 2;
end

[numFrames, numDims] = size(features);
delta = zeros(numFrames, numDims);
if numFrames == 0
    return;
end

deltaWindow = max(1, round(deltaWindow));
den = 2 * sum((1:deltaWindow) .^ 2);
padded = [repmat(features(1, :), deltaWindow, 1); ...
          features; ...
          repmat(features(end, :), deltaWindow, 1)];

for t = 1:numFrames
    acc = zeros(1, numDims);
    center = t + deltaWindow;
    for n = 1:deltaWindow
        acc = acc + n * (padded(center + n, :) - padded(center - n, :));
    end
    delta(t, :) = acc / den;
end

end
