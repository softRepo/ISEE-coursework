function dctMatrix = dct2_manual(numCoeffs, numInputs)
%DCT2_MANUAL 按 DCT-II 公式手工构造正交 DCT 矩阵。
% 第 k 个倒谱系数由所有 Mel 对数能量加权求和得到，
% 该变换将滤波器组能量从相关频带表示转换为较紧凑的倒谱表示。

if nargin < 2
    numInputs = numCoeffs;
end

dctMatrix = zeros(numCoeffs, numInputs);
n = 0:numInputs-1;

for k = 0:numCoeffs-1
    if k == 0
        alpha = sqrt(1 / numInputs);
    else
        alpha = sqrt(2 / numInputs);
    end
    dctMatrix(k + 1, :) = alpha * cos(pi * (2 * n + 1) * k / (2 * numInputs));
end

end
