function main_A_music_effect_analysis(rootDir, cfg)
%课题A：音乐合成与音效分析
%   读取《致爱丽丝》钢琴片段，构造单次回声和多延迟混响效果。
%   对原始信号和音效处理结果进行波形、FFT频谱和STFT时频分析。

if nargin < 1 || isempty(rootDir)
    rootDir = fileparts(fileparts(mfilename('fullpath')));
end
if nargin < 2 || isempty(cfg)
    addpath(rootDir);
    cfg = config_ABC_experiments();
end
addpath(fullfile(rootDir, 'common'));

inPath = fullfile(rootDir, 'segments', cfg.segments.A);
figDir = fullfile(rootDir, 'results', 'A_music_effect', 'figures');
audioDir = fullfile(rootDir, 'results', 'A_music_effect', 'audio');
tableDir = fullfile(rootDir, 'results', 'A_music_effect', 'tables');
reportFigDir = fullfile(rootDir, 'report', 'latex_report', 'figures');
reportTableDir = fullfile(rootDir, 'report', 'latex_report', 'tables');
ensure_dirs({figDir, audioDir, tableDir, reportFigDir, reportTableDir});

[x, fs, info] = load_audio_segment(inPath, cfg.targetFs); %#ok<ASGLU>
[x, ~] = normalize_audio_peak(x, 0.98);

% 单次回声模型：y[n] = x[n] + alpha*x[n-D]。
D = round(cfg.A.echoDelaySec * fs);
echo = x;
echo(D+1:end) = echo(D+1:end) + cfg.A.echoAlpha * x(1:end-D);
[echo, ~] = normalize_audio_peak(echo, 0.98);

% 多延迟混响模型：y[n] = x[n] + sum alpha_k*x[n-D_k]。
reverb = x;
for k = 1:numel(cfg.A.reverbDelaysSec)
    dk = round(cfg.A.reverbDelaysSec(k) * fs);
    if dk < numel(x)
        reverb(dk+1:end) = reverb(dk+1:end) + cfg.A.reverbGains(k) * x(1:end-dk);
    end
end
[reverb, ~] = normalize_audio_peak(reverb, 0.98);

audiowrite(fullfile(audioDir, 'A_original.wav'), x, fs);
audiowrite(fullfile(audioDir, 'A_echo.wav'), echo, fs);
audiowrite(fullfile(audioDir, 'A_reverb.wav'), reverb, fs);

% 保存原始钢琴片段的时域、频域和时频综合分析图。
plot_time_frequency_overview(x, fs, '课题A 原始钢琴片段时频分析', ...
    fullfile(figDir, 'A_original_waveform_spectrum.png'), cfg);

plot_echo_reverb_waveform(x, echo, reverb, fs, cfg, fullfile(figDir, 'A_echo_reverb_waveform_compare.png'));
plot_fft_compare(x, echo, reverb, fs, cfg, fullfile(figDir, 'A_fft_compare_original_echo_reverb.png'));
plot_stft_one(x, fs, cfg.A.stftShortWinMs, cfg, '原始钢琴片段STFT语谱图', fullfile(figDir, 'A_stft_original.png'));
plot_stft_one(echo, fs, cfg.A.stftShortWinMs, cfg, '单次回声音频STFT语谱图', fullfile(figDir, 'A_stft_echo.png'));
plot_stft_one(reverb, fs, cfg.A.stftShortWinMs, cfg, '多重混响音频STFT语谱图', fullfile(figDir, 'A_stft_reverb.png'));
plot_stft_window_compare(x, fs, cfg, fullfile(figDir, 'A_stft_window_compare.png'));

signals = {x, echo, reverb};
names = {'原始', '单次回声', '多重混响'};
notes = {
    '钢琴原始片段，用于基准FFT与STFT分析'
    sprintf('延迟%.2f秒、衰减系数%.2f的单次回声', cfg.A.echoDelaySec, cfg.A.echoAlpha)
    '多延迟叠加混响，模拟早期反射和空间拖尾'
    };
rows = cell(3, 7);
for i = 1:3
    % 计算音效处理前后的基础指标和主要频率范围。
    m = audio_metrics(signals{i}, fs);
    bw = estimate_effective_bandwidth(signals{i}, fs, 0.99, 40);
    rows(i, :) = {names{i}, m.duration, m.peak, m.rms, m.energy, ...
        sprintf('0-%.0f Hz', bw.fEffective), notes{i}};
end
T = cell2table(rows, 'VariableNames', {'处理类型','时长','峰值','RMS','能量','主要频率范围','说明'});
write_table_utf8(T, fullfile(tableDir, 'A_effect_metrics.csv'));
write_latex_table({'处理类型','时长/s','峰值','RMS','有效频率','说明'}, ...
    rows(:, [1 2 3 4 6 7]), fullfile(reportTableDir, 'A_effect_metrics_table.tex'));

copy_outputs(figDir, reportFigDir, {'A_original_waveform_spectrum.png','A_echo_reverb_waveform_compare.png', ...
    'A_fft_compare_original_echo_reverb.png','A_stft_original.png','A_stft_echo.png', ...
    'A_stft_reverb.png','A_stft_window_compare.png'});

safe_sound_play(x, fs, cfg.enablePlayback, 8);
fprintf('课题A完成：原始/回声/混响音频与图表已生成。\n');
end

function ensure_dirs(dirs)
%ENSURE_DIRS 创建课题A所需的结果输出目录
for i = 1:numel(dirs)
    if ~exist(dirs{i}, 'dir')
        mkdir(dirs{i});
    end
end
end

function plot_echo_reverb_waveform(x, echo, reverb, fs, cfg, outputPath)
%PLOT_ECHO_REVERB_WAVEFORM 绘制原始、回声和混响信号的波形对比
visibleState = ternary(cfg.showFigures, 'on', 'off');
fig = figure('Color','w','Visible',visibleState,'Position',[80,80,1200,760]);
t = (0:numel(x)-1)'/fs;
nLocal = min(numel(x), round(8*fs));
tiledlayout(fig, 2, 1, 'TileSpacing','compact','Padding','compact');
nexttile;
plot(t, x, 'k'); hold on; plot(t, echo, 'b'); plot(t, reverb, 'r');
grid on; xlabel('时间 / 秒'); ylabel('幅值'); title('完整波形对比');
legend('原始','回声','混响');
nexttile;
plot(t(1:nLocal), x(1:nLocal), 'k'); hold on; plot(t(1:nLocal), echo(1:nLocal), 'b'); plot(t(1:nLocal), reverb(1:nLocal), 'r');
grid on; xlabel('时间 / 秒'); ylabel('幅值'); title('前8秒局部波形对比');
legend('原始','回声','混响');
save_figure_safely(fig, outputPath);
if ~cfg.showFigures, close(fig); end
end

function plot_fft_compare(x, echo, reverb, fs, cfg, outputPath)
%PLOT_FFT_COMPARE 绘制原始、回声和混响信号的FFT频谱对比
visibleState = ternary(cfg.showFigures, 'on', 'off');
fig = figure('Color','w','Visible',visibleState,'Position',[80,80,1200,760]);
signals = {x, echo, reverb};
names = {'原始','回声','混响'};
colors = {'k','b','r'};
tiledlayout(fig, 2, 1, 'TileSpacing','compact','Padding','compact');
nexttile; hold on;
for i = 1:3
    s = compute_fft_spectrum(signals{i}, fs);
    mask = s.freq <= cfg.maxSpectrumFreqHz;
    plot(s.freq(mask), s.amp(mask), colors{i});
end
grid on; xlabel('频率 / Hz'); ylabel('幅度'); title('FFT幅度谱对比'); legend(names);
nexttile; hold on;
for i = 1:3
    s = compute_fft_spectrum(signals{i}, fs);
    mask = s.freq <= cfg.maxSpectrumFreqHz;
    plot(s.freq(mask), s.db(mask), colors{i});
end
grid on; xlabel('频率 / Hz'); ylabel('幅度 / dB'); title('dB频谱对比'); legend(names);
save_figure_safely(fig, outputPath);
if ~cfg.showFigures, close(fig); end
end

function plot_stft_one(x, fs, winMs, cfg, titleText, outputPath)
%PLOT_STFT_ONE 绘制指定窗长下的STFT语谱图
visibleState = ternary(cfg.showFigures, 'on', 'off');
st = compute_stft_manual(x, fs, winMs, cfg.A.stftHopRatio);
mask = st.freq <= cfg.maxSpectrumFreqHz;
fig = figure('Color','w','Visible',visibleState,'Position',[80,80,1100,650]);
imagesc(st.time, st.freq(mask), st.magDb(mask,:));
axis xy; colorbar; colormap turbo;
xlabel('时间 / 秒'); ylabel('频率 / Hz'); title(titleText);
save_figure_safely(fig, outputPath);
if ~cfg.showFigures, close(fig); end
end

function plot_stft_window_compare(x, fs, cfg, outputPath)
%PLOT_STFT_WINDOW_COMPARE 比较短窗和长窗STFT的时频分辨率差异
visibleState = ternary(cfg.showFigures, 'on', 'off');
stShort = compute_stft_manual(x, fs, cfg.A.stftShortWinMs, cfg.A.stftHopRatio);
stLong = compute_stft_manual(x, fs, cfg.A.stftLongWinMs, cfg.A.stftHopRatio);
maskS = stShort.freq <= cfg.maxSpectrumFreqHz;
maskL = stLong.freq <= cfg.maxSpectrumFreqHz;
fig = figure('Color','w','Visible',visibleState,'Position',[80,80,1200,760]);
tiledlayout(fig, 1, 2, 'TileSpacing','compact','Padding','compact');
nexttile; imagesc(stShort.time, stShort.freq(maskS), stShort.magDb(maskS,:)); axis xy; colorbar;
xlabel('时间 / 秒'); ylabel('频率 / Hz'); title(sprintf('短窗STFT：%d ms', cfg.A.stftShortWinMs));
nexttile; imagesc(stLong.time, stLong.freq(maskL), stLong.magDb(maskL,:)); axis xy; colorbar;
xlabel('时间 / 秒'); ylabel('频率 / Hz'); title(sprintf('长窗STFT：%d ms', cfg.A.stftLongWinMs));
save_figure_safely(fig, outputPath);
if ~cfg.showFigures, close(fig); end
end

function copy_outputs(srcDir, dstDir, names)
%COPY_OUTPUTS 将指定图像复制到配套材料目录
for i = 1:numel(names)
    copyfile(fullfile(srcDir, names{i}), fullfile(dstDir, names{i}));
end
end

function y = ternary(cond, a, b)
%TERNARY 根据逻辑条件返回两个候选值之一
if cond, y = a; else, y = b; end
end
