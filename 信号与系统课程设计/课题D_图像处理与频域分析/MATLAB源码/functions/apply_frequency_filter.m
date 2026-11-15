function [filtered_image, G_shift, imag_max] = apply_frequency_filter(F_shift, H)
%APPLY_FREQUENCY_FILTER 在频域完成二维滤波并返回空间域滤波结果。
%
% 输入：
%   F_shift - 经过 fftshift 后的图像频谱。
%   H       - 与 F_shift 尺寸相同的频域滤波器。
%
% 输出：
%   filtered_image - 逆变换得到的空间域实部图像。
%   G_shift        - 频域相乘后的中心化频谱，即 F_shift .* H。
%   imag_max       - ifft2 结果中虚部绝对值的最大值。
%
% 原理说明：
%   核心频域滤波步骤为：
%       G_shift = F_shift .* H;
%       g = ifft2(ifftshift(G_shift));
%       filtered_image = real(g);
%   对实图像和实对称滤波器，理论输出应为实数。实际计算中由于
%   浮点舍入误差，ifft2 可能产生极小虚部，因此记录虚部并取其实部。

if ~isequal(size(F_shift), size(H))
    error('频谱和滤波器尺寸不一致，无法进行频域相乘。');
end

G_shift = F_shift .* H;
filtered_complex = ifft2(ifftshift(G_shift));
imag_max = max(abs(imag(filtered_complex(:))));
filtered_image = real(filtered_complex);
end
