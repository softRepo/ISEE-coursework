function [H_LP, H_HP, U, V, D] = gaussian_frequency_filters(M, N, sigma)
%GAUSSIAN_FREQUENCY_FILTERS 构造二维高斯低通和高通频域滤波器。
%
% 输入：
%   M     - 图像行数。
%   N     - 图像列数。
%   sigma - 频域高斯滤波器标准差，控制频率保留范围。
%
% 输出：
%   H_LP - 二维高斯低通滤波器。
%   H_HP - 二维高斯高通滤波器，H_HP = 1 - H_LP。
%   U,V  - 以频谱中心为原点的二维频率坐标网格。
%   D    - 每个频率点到中心的距离。
%
% 原理说明：
%   fftshift 后零频分量位于频谱中心，因此坐标网格也以中心为原点。
%   高斯低通滤波器公式为：
%       H_LP(u,v) = exp( -D(u,v)^2 / (2*sigma^2) )
%   其中 D(u,v) = sqrt(u^2 + v^2)。
%   高斯高通滤波器可以由频率互补关系得到：
%       H_HP(u,v) = 1 - H_LP(u,v)

if nargin < 3
    error('必须提供 M、N 和 sigma。');
end
if M <= 0 || N <= 0 || sigma <= 0 || ~isfinite(sigma)
    error('M、N 必须为正数，sigma 必须为正的有限数。');
end

u = -floor(N/2) : (ceil(N/2) - 1);
v = -floor(M/2) : (ceil(M/2) - 1);
[U, V] = meshgrid(u, v);

D = sqrt(U.^2 + V.^2);
H_LP = exp(-(D.^2) ./ (2 * sigma^2));
H_HP = 1 - H_LP;
end
