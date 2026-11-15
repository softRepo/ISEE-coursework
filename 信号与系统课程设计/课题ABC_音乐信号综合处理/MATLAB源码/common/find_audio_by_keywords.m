function [filePath, fileInfo, allAudioFiles, warningMessages] = find_audio_by_keywords(musicDir, keywords, supportedExtensions, taskName)
%FIND_AUDIO_BY_KEYWORDS 按关键词在music目录中匹配音频文件
%   keywords为候选关键词列表，supportedExtensions为允许的音频扩展名。
%   如果匹配失败，函数给出中文错误提示并列出已发现的音频文件。

warningMessages = {};

if ~exist(musicDir, 'dir')
    error('未找到音频目录：%s', musicDir);
end

entries = dir(musicDir);
isAudio = false(size(entries));
for i = 1:numel(entries)
    if entries(i).isdir
        continue;
    end
    [~, ~, ext] = fileparts(entries(i).name);
    isAudio(i) = any(strcmpi(ext, supportedExtensions));
end
allAudioFiles = entries(isAudio);

if isempty(allAudioFiles)
    error('music目录中未找到支持的音频文件。目录：%s', musicDir);
end

isMatch = false(size(allAudioFiles));
for iFile = 1:numel(allAudioFiles)
    % 逐个检查文件名是否包含任一关键词。
    fileName = string(allAudioFiles(iFile).name);
    for iKey = 1:numel(keywords)
        if contains(fileName, string(keywords{iKey}), 'IgnoreCase', true)
            isMatch(iFile) = true;
            break;
        end
    end
end

matchedFiles = allAudioFiles(isMatch);
if isempty(matchedFiles)
    foundNames = join("  - " + string({allAudioFiles.name})', newline);
    keyText = strjoin(keywords, ' / ');
    error(['%s 未能根据关键词匹配到音频文件。', newline, ...
        '关键词：%s', newline, ...
        'music目录下找到的音频文件：', newline, '%s'], ...
        taskName, keyText, char(foundNames));
end

if numel(matchedFiles) > 1
    names = strjoin({matchedFiles.name}, '；');
    warningMessages{end + 1} = sprintf('%s 匹配到多个文件，默认使用第一个：%s。全部匹配：%s', ...
        taskName, matchedFiles(1).name, names);
end

fileInfo = matchedFiles(1);
filePath = fullfile(fileInfo.folder, fileInfo.name);
end
