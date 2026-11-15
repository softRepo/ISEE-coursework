function mel = hz2mel_local(hz)
%HZ2MEL_LOCAL Hz 到 Mel 频率的转换。
% 转换公式：m = 2595 log10(1 + f/700)。

mel = 2595 .* log10(1 + hz ./ 700);

end
