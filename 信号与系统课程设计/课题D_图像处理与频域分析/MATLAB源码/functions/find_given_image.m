function image_path = find_given_image(search_dir, base_name)
%FIND_GIVEN_IMAGE 查找指定主名的图像文件。
%
% 输入：
%   search_dir - 查找目录，通常为课题D目录。
%   base_name  - 不含扩展名的图像文件名，例如“课题D给定图像”。
%
% 输出：
%   image_path - 找到的图像完整路径；若未找到则返回空字符串。
%
% 原理说明：
%   本函数只匹配给定的文件主名，并在常见图像扩展名中查找，
%   包括 .png、.jpg、.jpeg、.bmp、.tif、.tiff，用于避免把
%   其他课题或中间结果误认为输入图像。

image_path = '';

if nargin < 2 || isempty(base_name)
    base_name = '课题D给定图像';
end
if nargin < 1 || isempty(search_dir)
    search_dir = pwd;
end
if exist(search_dir, 'dir') ~= 7
    return;
end

valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'};
items = dir(search_dir);
candidates = {};

for k = 1:numel(items)
    if items(k).isdir
        continue;
    end
    [~, name_part, ext_part] = fileparts(items(k).name);
    if strcmp(name_part, base_name) && any(strcmpi(ext_part, valid_extensions))
        candidates{end + 1} = fullfile(search_dir, items(k).name); %#ok<AGROW>
    end
end

if isempty(candidates)
    return;
end

if numel(candidates) > 1
    warning('找到多个“%s”图像文件，将使用第一个：%s', base_name, candidates{1});
end

image_path = candidates{1};
end
