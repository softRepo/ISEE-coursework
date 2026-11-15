function safe_sound_play(x, fs, enablePlayback, previewSeconds)
%SAFE_SOUND_PLAY 根据配置播放短音频预览
%   enablePlayback为false时不播放。
%   previewSeconds用于限制播放时长，避免连续播放完整音频。

if nargin < 4 || isempty(previewSeconds)
    previewSeconds = 8;
end
if ~enablePlayback
    return;
end
n = min(numel(x), round(previewSeconds * fs));
sound(x(1:n), fs);
pause(n / fs + 0.2);
end
