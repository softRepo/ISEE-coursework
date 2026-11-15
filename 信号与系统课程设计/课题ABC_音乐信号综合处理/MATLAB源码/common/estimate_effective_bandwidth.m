function bw = estimate_effective_bandwidth(x, fs, energyRatio, dbDrop)
%ESTIMATE_EFFECTIVE_BANDWIDTH 基于累计频谱能量和dB阈值估计有效最高频率
%   energyRatio用于累计能量阈值，dbDrop用于主要频率成分的dB阈值。
%   输出结果为工程意义上的有效带宽估计，不直接等同于Nyquist频率。

if nargin < 3 || isempty(energyRatio)
    energyRatio = 0.99;
end
if nargin < 4 || isempty(dbDrop)
    dbDrop = 40;
end

spec = compute_fft_spectrum(x, fs);
freq = spec.freq;
power = spec.power;
totalPower = sum(power);
if totalPower <= eps
    fEnergy = 0;
else
    % 查找累计频谱能量首次达到指定比例的位置。
    cumulative = cumsum(power) / totalPower;
    idx = find(cumulative >= energyRatio, 1, 'first');
    fEnergy = freq(idx);
end

dbThreshold = max(spec.db) - dbDrop;
idxDb = find(spec.db >= dbThreshold);
if isempty(idxDb)
    fDb = fEnergy;
else
    fDb = max(freq(idxDb));
end

bw = struct();
bw.fEnergy = fEnergy;
bw.fDb = fDb;
bw.fEffective = min(max(fEnergy, 20), fs / 2);
bw.fsMin = 2 * bw.fEffective;
bw.energyRatio = energyRatio;
bw.dbDrop = dbDrop;
bw.freq = freq;
bw.amp = spec.amp;
bw.db = spec.db;
bw.cumulativeEnergy = cumsum(power) / max(totalPower, eps);
end
