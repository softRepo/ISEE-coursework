function hz = mel2hz_local(mel)
%MEL2HZ_LOCAL Mel 到 Hz 的逆变换。
% 逆变换公式：f = 700 (10^(m/2595) - 1)。

hz = 700 .* (10 .^ (mel ./ 2595) - 1);

end
