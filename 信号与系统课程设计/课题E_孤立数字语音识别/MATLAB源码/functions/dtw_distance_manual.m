function [distance, info] = dtw_distance_manual(X, Y, cfg, needDetails)
%DTW_DISTANCE_MANUAL 手工计算 DTW 动态时间规整距离，不调用 MATLAB dtw 函数。
%
% 递推公式：
% D(i,j) = d(i,j) + min(D(i-1,j), D(i,j-1), D(i-1,j-1))
% 其中 d(i,j) 是第一个语音第 i 帧与第二个语音第 j 帧的局部距离；
% D(i,j) 是从起点到达该点的最小累积距离。
% DTW 允许时间轴局部伸缩，因此可以缓解不同说话速度导致的语音长短差异。
% 对长度分别为 N 和 M 的两段语音，标准 DTW 计算复杂度约为 O(NM)。

if nargin < 4 || isempty(needDetails)
    needDetails = true;
end

info = struct();
if isempty(X) || isempty(Y)
    distance = inf;
    info.warning = '输入特征为空。';
    return;
end
if size(X, 2) ~= size(Y, 2)
    error('DTW:DimMismatch', '两个特征矩阵的列数不一致，无法计算 DTW。');
end
if any(~isfinite(X(:))) || any(~isfinite(Y(:)))
    distance = inf;
    info.warning = '输入特征包含 NaN 或 Inf。';
    return;
end

X = double(X);
Y = double(Y);
N = size(X, 1);
M = size(Y, 1);
localDist = compute_local_distance(X, Y, cfg.dtwDistanceMetric);

if cfg.useSakoeChibaBand
    band = max(abs(N - M), round(cfg.sakoeChibaRatio * max(N, M)));
else
    band = inf;
end

if ~needDetails
    [distance, pathLength] = dtw_distance_two_rows(localDist, band);
    info.pathLength = pathLength;
    info.warning = '';
    return;
end

accDist = inf(N, M);
pathLen = inf(N, M);
traceBack = zeros(N, M, 'uint8'); % 路径来源编码：1=上，2=左，3=左上。
accDist(1, 1) = localDist(1, 1);
pathLen(1, 1) = 1;

for i = 1:N
    jStart = max(1, i - band);
    jEnd = min(M, i + band);
    for j = jStart:jEnd
        if i == 1 && j == 1
            continue;
        end

        candidates = [inf inf inf];
        lens = [inf inf inf];
        if i > 1
            candidates(1) = accDist(i - 1, j);
            lens(1) = pathLen(i - 1, j);
        end
        if j > 1
            candidates(2) = accDist(i, j - 1);
            lens(2) = pathLen(i, j - 1);
        end
        if i > 1 && j > 1
            candidates(3) = accDist(i - 1, j - 1);
            lens(3) = pathLen(i - 1, j - 1);
        end

        [bestPrev, bestIdx] = min(candidates);
        if isfinite(bestPrev)
            accDist(i, j) = localDist(i, j) + bestPrev;
            pathLen(i, j) = lens(bestIdx) + 1;
            traceBack(i, j) = uint8(bestIdx);
        end
    end
end

if ~isfinite(accDist(N, M))
    distance = inf;
    info.localDistance = localDist;
    info.accumulatedDistance = accDist;
    info.path = zeros(0, 2);
    info.pathLength = inf;
    info.rawDistance = inf;
    info.warning = 'DTW 路径不可达，请检查 Sakoe-Chiba 带宽。';
    return;
end

path = backtrace_path(traceBack, N, M);
distance = accDist(N, M) / max(size(path, 1), 1);

info.localDistance = localDist;
info.accumulatedDistance = accDist;
info.path = path;
info.pathLength = size(path, 1);
info.rawDistance = accDist(N, M);
info.warning = '';

end

function localDist = compute_local_distance(X, Y, metric)
switch lower(metric)
    case 'cosine'
        xNorm = sqrt(sum(X .^ 2, 2));
        yNorm = sqrt(sum(Y .^ 2, 2));
        xNorm(xNorm < eps) = 1;
        yNorm(yNorm < eps) = 1;
        Xn = X ./ repmat(xNorm, 1, size(X, 2));
        Yn = Y ./ repmat(yNorm, 1, size(Y, 2));
        localDist = 1 - Xn * Yn.';
        localDist = max(localDist, 0);
    otherwise
        x2 = sum(X .^ 2, 2);
        y2 = sum(Y .^ 2, 2).';
        dist2 = bsxfun(@plus, x2, y2) - 2 * (X * Y.');
        localDist = sqrt(max(dist2, 0));
end
localDist(~isfinite(localDist)) = inf;
end

function [distance, finalPathLength] = dtw_distance_two_rows(localDist, band)
[N, M] = size(localDist);
prev = inf(1, M);
curr = inf(1, M);
prevLen = inf(1, M);
currLen = inf(1, M);

for i = 1:N
    curr(:) = inf;
    currLen(:) = inf;
    jStart = max(1, i - band);
    jEnd = min(M, i + band);
    for j = jStart:jEnd
        if i == 1 && j == 1
            curr(j) = localDist(1, 1);
            currLen(j) = 1;
            continue;
        end

        candidates = [inf inf inf];
        lens = [inf inf inf];
        if i > 1
            candidates(1) = prev(j);
            lens(1) = prevLen(j);
        end
        if j > 1
            candidates(2) = curr(j - 1);
            lens(2) = currLen(j - 1);
        end
        if i > 1 && j > 1
            candidates(3) = prev(j - 1);
            lens(3) = prevLen(j - 1);
        end

        [bestPrev, bestIdx] = min(candidates);
        if isfinite(bestPrev)
            curr(j) = localDist(i, j) + bestPrev;
            currLen(j) = lens(bestIdx) + 1;
        end
    end
    prev = curr;
    prevLen = currLen;
end

if isfinite(prev(M))
    finalPathLength = prevLen(M);
    distance = prev(M) / max(finalPathLength, 1);
else
    finalPathLength = inf;
    distance = inf;
end
end

function path = backtrace_path(traceBack, i, j)
path = zeros(i + j, 2);
count = 1;
path(count, :) = [i j];

while i > 1 || j > 1
    move = traceBack(i, j);
    if move == 1
        i = i - 1;
    elseif move == 2
        j = j - 1;
    elseif move == 3
        i = i - 1;
        j = j - 1;
    else
        break;
    end
    count = count + 1;
    path(count, :) = [i j];
end

path = path(1:count, :);
path = flipud(path);
end
