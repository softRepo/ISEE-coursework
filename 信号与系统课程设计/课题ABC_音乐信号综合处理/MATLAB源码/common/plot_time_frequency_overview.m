function plot_time_frequency_overview(x, fs, titleText, outputPath, cfg)
%PLOT_TIME_FREQUENCY_OVERVIEW 绘制时域、频谱和STFT综合图
%   输入为单声道音频、采样率、图标题、输出路径和统一配置。
%   输出图像用于比较音频信号的时域、频域和时频特征。

visibleState = 'off';
if cfg.showFigures
    visibleState = 'on';
end
fig = figure('Color', 'w', 'Visible', visibleState, 'Position', [80, 80, 1200, 900]);
tl = tiledlayout(fig, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, titleText, 'Interpreter', 'none');

t = (0:numel(x)-1)' / fs;
nexttile(tl, [1, 2]);
plot(t, x, 'b');
grid on; xlabel('时间 / 秒'); ylabel('幅值'); title('完整时域波形');

nexttile(tl);
nLocal = min(numel(x), round(cfg.localWaveSeconds * fs));
plot(t(1:nLocal), x(1:nLocal), 'Color', [0.1, 0.5, 0.25]);
grid on; xlabel('时间 / 秒'); ylabel('幅值'); title('局部时域波形');

spec = compute_fft_spectrum(x, fs);
mask = spec.freq <= min(cfg.maxSpectrumFreqHz, fs/2);
nexttile(tl);
plot(spec.freq(mask), spec.amp(mask), 'r');
grid on; xlabel('频率 / Hz'); ylabel('幅度'); title('单边FFT幅度谱');

nexttile(tl);
plot(spec.freq(mask), spec.db(mask), 'Color', [0.4, 0.2, 0.65]);
grid on; xlabel('频率 / Hz'); ylabel('幅度 / dB'); title('dB频谱');

nexttile(tl);
stftData = compute_stft_manual(x, fs, 50, 0.5);
stftMask = stftData.freq <= min(cfg.maxSpectrumFreqHz, fs/2);
imagesc(stftData.time, stftData.freq(stftMask), stftData.magDb(stftMask, :));
axis xy; xlabel('时间 / 秒'); ylabel('频率 / Hz'); title('STFT语谱图');
colorbar;

if cfg.saveFigures
    save_figure_safely(fig, outputPath);
end
if ~cfg.showFigures
    close(fig);
end
end
