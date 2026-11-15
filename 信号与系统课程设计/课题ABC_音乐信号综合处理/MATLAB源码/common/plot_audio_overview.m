function fig = plot_audio_overview(audioData, fs, taskCfg, sourceFileName, cfg)
%PLOT_AUDIO_OVERVIEW 生成时域、频域和语谱图综合检查图
%   用于音频片段准备阶段的基础检查。
%   图像包含完整波形、局部波形、FFT频谱、dB频谱和语谱图。

if cfg.showFigures
    visibleState = 'on';
else
    visibleState = 'off';
end

audio = double(audioData(:));
audio(~isfinite(audio)) = 0;
n = numel(audio);
time = (0:n - 1)' / fs;

fig = figure('Name', [taskCfg.shortName, '_audio_overview'], ...
    'Color', 'w', 'Visible', visibleState, 'Position', [80, 80, 1300, 1000]);

layout = tiledlayout(fig, 4, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(layout, sprintf('%s：%s（%s）', taskCfg.topic, taskCfg.displayName, sourceFileName), ...
    'Interpreter', 'none', 'FontWeight', 'bold');

% 完整时域波形。长音频绘图时抽样显示，避免图像文件过大。
ax1 = nexttile(layout, [1, 2]);
maxPlotPoints = 200000;
if n > maxPlotPoints
    plotIndex = unique(round(linspace(1, n, maxPlotPoints)));
else
    plotIndex = 1:n;
end
plot(ax1, time(plotIndex), audio(plotIndex), 'Color', [0.05, 0.30, 0.60], 'LineWidth', 0.7);
grid(ax1, 'on');
xlabel(ax1, '时间 / 秒');
ylabel(ax1, '幅值');
title(ax1, '完整时域波形');
ylim(ax1, [-1.05, 1.05]);

% 前若干秒局部波形。
ax2 = nexttile(layout);
localSamples = min(n, round(cfg.localWaveSeconds * fs));
plot(ax2, time(1:localSamples), audio(1:localSamples), 'Color', [0.05, 0.45, 0.32], 'LineWidth', 0.7);
grid(ax2, 'on');
xlabel(ax2, '时间 / 秒');
ylabel(ax2, '幅值');
title(ax2, sprintf('前%.1f秒局部波形', localSamples / fs));
ylim(ax2, [-1.05, 1.05]);

% 单边FFT幅度谱。
[freq, mag, magDb] = plot_audio_spectrum(audio, fs);
freqLimit = min(cfg.fftMaxFreq, fs / 2);
freqMask = freq <= freqLimit;

ax3 = nexttile(layout);
plot(ax3, freq(freqMask), mag(freqMask), 'Color', [0.72, 0.22, 0.18], 'LineWidth', 0.8);
grid(ax3, 'on');
xlabel(ax3, '频率 / Hz');
ylabel(ax3, '幅度');
title(ax3, '单边FFT幅度谱');
xlim(ax3, [0, freqLimit]);

% dB频谱。
ax4 = nexttile(layout, [1, 2]);
plot(ax4, freq(freqMask), magDb(freqMask), 'Color', [0.39, 0.24, 0.62], 'LineWidth', 0.8);
grid(ax4, 'on');
xlabel(ax4, '频率 / Hz');
ylabel(ax4, '幅度 / dB');
title(ax4, 'dB频谱');
xlim(ax4, [0, freqLimit]);
ylim(ax4, [max(-160, max(magDb(freqMask)) - 120), max(magDb(freqMask)) + 5]);

% STFT或简易语谱图检查。
ax5 = nexttile(layout, [1, 2]);
[stftDb, stftFreq, stftTime] = compute_spectrogram_for_plot(audio, fs);
specLimit = min(cfg.spectrogramMaxFreq, fs / 2);
specMask = stftFreq <= specLimit;
imagesc(ax5, stftTime, stftFreq(specMask), stftDb(specMask, :));
axis(ax5, 'xy');
xlabel(ax5, '时间 / 秒');
ylabel(ax5, '频率 / Hz');
title(ax5, 'STFT语谱图检查');
ylim(ax5, [0, specLimit]);
colorbar(ax5);
try
    colormap(ax5, 'turbo');
catch
    colormap(ax5, 'parula');
end
end

function [specDb, freq, time] = compute_spectrogram_for_plot(audio, fs)
%COMPUTE_SPECTROGRAM_FOR_PLOT 优先使用spectrogram，缺失时手工分帧FFT
winLength = min(2048, max(256, 2 ^ nextpow2(round(0.046 * fs))));
winLength = min(winLength, numel(audio));
if winLength < 16
    winLength = numel(audio);
end
overlap = floor(0.75 * winLength);
nfft = max(2048, 2 ^ nextpow2(winLength));

sampleIndex = (0:winLength - 1)';
window = 0.54 - 0.46 * cos(2 * pi * sampleIndex / max(winLength - 1, 1));

if exist('spectrogram', 'file') == 2
    [spec, freq, time] = spectrogram(audio, window, overlap, nfft, fs);
else
    [spec, freq, time] = simple_spectrogram(audio, window, overlap, nfft, fs);
end

specDb = 20 * log10(abs(spec) + eps);
maxDb = max(specDb(:));
specDb = max(specDb, maxDb - 100);
end

function [spec, freq, time] = simple_spectrogram(audio, window, overlap, nfft, fs)
%SIMPLE_SPECTROGRAM 无工具箱时使用的简易分帧FFT
winLength = numel(window);
hop = max(1, winLength - overlap);
if numel(audio) < winLength
    audio = [audio; zeros(winLength - numel(audio), 1)];
end

frameCount = 1 + floor((numel(audio) - winLength) / hop);
halfCount = floor(nfft / 2) + 1;
spec = zeros(halfCount, frameCount);

for iFrame = 1:frameCount
    startIndex = (iFrame - 1) * hop + 1;
    frame = audio(startIndex:startIndex + winLength - 1) .* window;
    frameSpec = fft(frame, nfft);
    spec(:, iFrame) = frameSpec(1:halfCount);
end

freq = (0:halfCount - 1)' * fs / nfft;
time = ((0:frameCount - 1) * hop + winLength / 2) / fs;
end
