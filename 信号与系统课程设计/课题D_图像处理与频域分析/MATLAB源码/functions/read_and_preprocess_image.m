function [original_display, gray_image, info] = read_and_preprocess_image(image_path)
%READ_AND_PREPROCESS_IMAGE 读取图像并完成灰度化与归一化处理。
%
% 输入：
%   image_path - 给定图像的完整路径。
%
% 输出：
%   original_display - 用于显示的原始图像，RGB 或灰度，double 且范围为 [0,1]。
%   gray_image       - 灰度归一化图像，double 且范围为 [0,1]。
%   info             - 图像信息结构体，包括文件名、原始类型、灰度转换方式等。
%
% 原理说明：
%   若输入是彩色图像，优先调用 rgb2gray；若当前 MATLAB 环境没有
%   rgb2gray 或调用失败，则使用手动加权公式：
%   Gray = 0.2989*R + 0.5870*G + 0.1140*B。
%   归一化后图像进入 fft2 进行频域处理。

if nargin < 1 || isempty(image_path)
    error('必须提供图像路径。');
end
if exist(image_path, 'file') ~= 2
    error('图像文件不存在：%s', image_path);
end

[raw_image, color_map] = imread(image_path);
[~, file_name, file_ext] = fileparts(image_path);

info = struct();
info.file_name = [file_name, file_ext];
info.original_class = class(raw_image);
info.original_size = size(raw_image);
info.is_indexed = ~isempty(color_map);
info.is_color = false;
info.gray_method = '原始灰度图像归一化';

if ~isempty(color_map)
    rgb_image = indexed_to_rgb(raw_image, color_map);
    original_display = to_unit_double(rgb_image);
    [gray_image, method] = rgb_to_gray_compatible(rgb_image);
    info.is_color = true;
    info.gray_method = ['索引图像转RGB后', method];
elseif ndims(raw_image) == 3 && size(raw_image, 3) >= 3
    rgb_image = raw_image(:, :, 1:3);
    original_display = to_unit_double(rgb_image);
    [gray_image, method] = rgb_to_gray_compatible(rgb_image);
    info.is_color = true;
    info.gray_method = method;
else
    gray_image = to_unit_double(raw_image);
    original_display = gray_image;
end

gray_image = normalize_for_display(gray_image, 'clip01');
original_display = normalize_for_display(original_display, 'clip01');
end

function [gray_image, method] = rgb_to_gray_compatible(rgb_image)
%RGB_TO_GRAY_COMPATIBLE 优先使用 rgb2gray，失败时使用手动加权公式。

method = 'rgb2gray';
if exist('rgb2gray', 'file') == 2 || exist('rgb2gray', 'builtin') == 5
    try
        gray_candidate = rgb2gray(rgb_image);
        gray_image = to_unit_double(gray_candidate);
        return;
    catch
        % 当 rgb2gray 对当前数据类型或运行环境不可用时，使用手动公式。
    end
end

rgb_double = to_unit_double(rgb_image);
gray_image = 0.2989 * rgb_double(:, :, 1) + ...
             0.5870 * rgb_double(:, :, 2) + ...
             0.1140 * rgb_double(:, :, 3);
method = '手动加权灰度公式';
end

function rgb_image = indexed_to_rgb(indexed_image, color_map)
%INDEXED_TO_RGB 将索引图像转换为 RGB 图像。

if exist('ind2rgb', 'file') == 2 || exist('ind2rgb', 'builtin') == 5
    try
        rgb_image = ind2rgb(indexed_image, color_map);
        return;
    catch
        % 当 ind2rgb 不可用时，使用后续兼容实现。
    end
end

idx = double(indexed_image);
if isinteger(indexed_image) || islogical(indexed_image)
    idx = idx + 1;
end
idx(idx < 1) = 1;
idx(idx > size(color_map, 1)) = size(color_map, 1);

rows = size(indexed_image, 1);
cols = size(indexed_image, 2);
rgb_image = zeros(rows, cols, 3);
rgb_image(:, :, 1) = reshape(color_map(idx(:), 1), rows, cols);
rgb_image(:, :, 2) = reshape(color_map(idx(:), 2), rows, cols);
rgb_image(:, :, 3) = reshape(color_map(idx(:), 3), rows, cols);
end

function Y = to_unit_double(X)
%TO_UNIT_DOUBLE 将常见图像数据类型转换为 [0,1] 范围内的 double。

if isfloat(X)
    Y = double(X);
    finite_values = Y(isfinite(Y));
    if isempty(finite_values)
        Y = zeros(size(Y));
        return;
    end
    min_value = min(finite_values(:));
    max_value = max(finite_values(:));
    if min_value < 0 || max_value > 1
        if max_value > min_value
            Y = (Y - min_value) ./ (max_value - min_value);
        else
            Y = zeros(size(Y));
        end
    end
elseif islogical(X)
    Y = double(X);
elseif isa(X, 'uint8')
    Y = double(X) ./ 255;
elseif isa(X, 'uint16')
    Y = double(X) ./ 65535;
elseif isa(X, 'uint32')
    Y = double(X) ./ double(intmax('uint32'));
elseif isa(X, 'uint64')
    Y = double(X) ./ double(intmax('uint64'));
else
    Y = double(X);
    min_value = min(Y(:));
    max_value = max(Y(:));
    if max_value > min_value
        Y = (Y - min_value) ./ (max_value - min_value);
    else
        Y = zeros(size(Y));
    end
end

Y(~isfinite(Y)) = 0;
Y(Y < 0) = 0;
Y(Y > 1) = 1;
end
