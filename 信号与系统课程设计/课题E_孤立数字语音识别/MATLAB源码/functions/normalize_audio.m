function y = normalize_audio(x)
%NORMALIZE_AUDIO 对语音信号进行幅值归一化，减小录音幅度差异的影响。

x = double(x(:));
x(~isfinite(x)) = 0;
peak = max(abs(x));

if isempty(peak) || peak < eps
    y = zeros(size(x));
else
    y = x ./ peak;
end

end
