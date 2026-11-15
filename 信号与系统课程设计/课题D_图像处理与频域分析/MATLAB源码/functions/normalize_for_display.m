function Y = normalize_for_display(X, mode)
%NORMALIZE_FOR_DISPLAY 将矩阵转换为适合图像显示或保存的 [0,1] 数据。
%
% 输入：
%   X    - 输入矩阵或图像，可为实数或复数。
%   mode - 归一化模式：
%          'clip01'：直接截断到 [0,1]，适合已经归一化的灰度图；
%          'minmax'：按最小值和最大值线性拉伸，适合频谱和高通结果。
%
% 输出：
%   Y - double 类型，范围为 [0,1]。
%
% 原理说明：
%   高通滤波结果通常包含正负值，直接显示不便观察，因此常用 minmax
%   拉伸进行可视化。该处理只影响显示和保存，不改变主脚本中统计的
%   原始滤波结果。

if nargin < 2 || isempty(mode)
    mode = 'minmax';
end

Y = double(real(X));
Y(~isfinite(Y)) = 0;

switch lower(mode)
    case {'clip01', 'clip'}
        Y(Y < 0) = 0;
        Y(Y > 1) = 1;
    otherwise
        if isempty(Y)
            return;
        end
        min_value = min(Y(:));
        max_value = max(Y(:));
        if max_value > min_value
            Y = (Y - min_value) ./ (max_value - min_value);
        else
            Y = zeros(size(Y));
        end
        Y(Y < 0) = 0;
        Y(Y > 1) = 1;
end
end
