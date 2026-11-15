function [segment, actualStartTime, actualDuration, warningMessages] = extract_audio_segment(audioData, fs, startTime, duration, minDuration, taskName)
%EXTRACT_AUDIO_SEGMENT 按秒截取音频片段，并在越界时自动调整
%   startTime和duration均以秒为单位。
%   当指定范围超出音频长度时，函数尽量保持片段长度并返回警告信息。

warningMessages = {};

if isempty(audioData)
    error('%s 原始音频为空，无法截取片段。', taskName);
end

totalDuration = size(audioData, 1) / fs;
if totalDuration < minDuration
    error('%s 原始音频总时长仅 %.3f 秒，不足 %.3f 秒，不能满足课程片段要求。', ...
        taskName, totalDuration, minDuration);
end

requestedStart = max(0, startTime);
requestedDuration = max(0, duration);
if requestedDuration <= 0
    error('%s 配置的截取时长必须大于0秒。', taskName);
end

actualStartTime = requestedStart;
if requestedStart >= totalDuration
    % 开始时间超出音频末尾时，将截取起点前移到可用范围内。
    actualStartTime = max(0, totalDuration - requestedDuration);
    warningMessages{end + 1} = sprintf('%s 配置的开始时间 %.3f 秒超出音频总时长 %.3f 秒，已自动调整为 %.3f 秒。', ...
        taskName, requestedStart, totalDuration, actualStartTime);
end

if actualStartTime + requestedDuration > totalDuration
    % 截取终点超出音频末尾时，根据总时长调整截取范围。
    if totalDuration >= requestedDuration
        newStart = max(0, totalDuration - requestedDuration);
        warningMessages{end + 1} = sprintf('%s 配置的截取范围 %.3f-%.3f 秒超出音频末尾，已将开始时间调整为 %.3f 秒以尽量保持 %.3f 秒片段。', ...
            taskName, actualStartTime, actualStartTime + requestedDuration, newStart, requestedDuration);
        actualStartTime = newStart;
    else
        warningMessages{end + 1} = sprintf('%s 原始音频时长 %.3f 秒不足配置片段 %.3f 秒，已从开头截取到末尾。', ...
            taskName, totalDuration, requestedDuration);
        actualStartTime = 0;
    end
end

startSample = floor(actualStartTime * fs) + 1;
endSample = min(size(audioData, 1), startSample + round(requestedDuration * fs) - 1);
if endSample < startSample
    error('%s 截取范围无有效采样点，请检查配置。', taskName);
end

segment = audioData(startSample:endSample, :);
actualStartTime = (startSample - 1) / fs;
actualDuration = size(segment, 1) / fs;

if actualDuration < minDuration
    error('%s 实际截取片段仅 %.3f 秒，不足 %.3f 秒，请更换音频或调整截取参数。', ...
        taskName, actualDuration, minDuration);
end
end
