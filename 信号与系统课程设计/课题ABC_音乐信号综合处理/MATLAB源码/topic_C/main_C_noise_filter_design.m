function main_C_noise_filter_design(rootDir, cfg)
%课题C：音频加噪与滤波器设计
%   读取 Faded 音频片段，叠加白噪声和单频噪声。
%   设计低通滤波器、陷波滤波器和组合滤波流程，并比较去噪指标。

if nargin < 1 || isempty(rootDir)
    rootDir = fileparts(fileparts(mfilename('fullpath')));
end
if nargin < 2 || isempty(cfg)
    addpath(rootDir);
    cfg = config_ABC_experiments();
end
addpath(fullfile(rootDir, 'common'));
rng(cfg.rngSeed);

inPath = fullfile(rootDir, 'segments', cfg.segments.C);
figDir = fullfile(rootDir, 'results', 'C_denoising', 'figures');
audioDir = fullfile(rootDir, 'results', 'C_denoising', 'audio');
tableDir = fullfile(rootDir, 'results', 'C_denoising', 'tables');
reportFigDir = fullfile(rootDir, 'report', 'latex_report', 'figures');
reportTableDir = fullfile(rootDir, 'report', 'latex_report', 'tables');
ensure_dirs({figDir, audioDir, tableDir, reportFigDir, reportTableDir});

[x, fs] = load_audio_segment(inPath, cfg.targetFs);
[x, ~] = normalize_audio_peak(x, 0.98);
t = (0:numel(x)-1)' / fs;

% 检查单频噪声频率是否满足奈奎斯特频率限制。
if cfg.C.noiseFreqHz >= fs/2
    error('单频噪声频率 %.1f Hz 超过Nyquist频率 %.1f Hz。', cfg.C.noiseFreqHz, fs/2);
end

signalPower = mean(x.^2);
noisePower = signalPower / 10^(cfg.C.whiteNoiseSNRdB/10);
whiteNoise = sqrt(noisePower) * randn(size(x));
toneNoise = cfg.C.singleToneNoiseAmp * sin(2*pi*cfg.C.noiseFreqHz*t);

% 构造白噪声、单频噪声和混合噪声三类测试信号。
xWhite = normalize_audio_peak(x + whiteNoise, 0.98);
xTone = normalize_audio_peak(x + toneNoise, 0.98);
xMixed = normalize_audio_peak(x + whiteNoise + toneNoise, 0.98);

[bLp, aLp] = design_lowpass(cfg.C.defaultLowpassOrder, cfg.C.defaultLowpassCutoffHz, fs);
[bNotch, aNotch] = design_notch(cfg.C.noiseFreqHz, cfg.C.notchBandwidthHz, fs);

% 分别进行低通滤波、陷波滤波和低通加陷波组合滤波。
yLow = normalize_audio_peak(apply_filter_safe(bLp, aLp, xWhite), 0.98);
yNotch = normalize_audio_peak(apply_filter_safe(bNotch, aNotch, xTone), 0.98);
yCombined = normalize_audio_peak(apply_filter_safe(bNotch, aNotch, apply_filter_safe(bLp, aLp, xMixed)), 0.98);

audiowrite(fullfile(audioDir, 'C_original.wav'), x, fs);
audiowrite(fullfile(audioDir, 'C_white_noise.wav'), xWhite, fs);
audiowrite(fullfile(audioDir, 'C_tone_noise.wav'), xTone, fs);
audiowrite(fullfile(audioDir, 'C_mixed_noise.wav'), xMixed, fs);
audiowrite(fullfile(audioDir, 'C_denoised_lowpass.wav'), yLow, fs);
audiowrite(fullfile(audioDir, 'C_denoised_notch.wav'), yNotch, fs);
audiowrite(fullfile(audioDir, 'C_denoised_combined.wav'), yCombined, fs);

plot_noise_compare(x, xWhite, xTone, xMixed, fs, cfg, fullfile(figDir, 'C_noise_waveform_spectrum_compare.png'));
plot_filter_response(bLp, aLp, bNotch, aNotch, fs, cfg, fullfile(figDir, 'C_filter_frequency_response.png'));
plot_denoise_waveform(x, xMixed, yLow, yNotch, yCombined, fs, cfg, fullfile(figDir, 'C_denoise_waveform_compare.png'));
plot_denoise_spectrum(x, xMixed, yLow, yNotch, yCombined, fs, cfg, fullfile(figDir, 'C_denoise_spectrum_compare.png'));

rows = {};
% 计算滤波前后的信噪比和均方根误差。
snrWhiteBefore = audio_metrics(xWhite, fs, x).snr;
snrToneBefore = audio_metrics(xTone, fs, x).snr;
snrMixedBefore = audio_metrics(xMixed, fs, x).snr;
mLow = audio_metrics(yLow, fs, x);
mNotch = audio_metrics(yNotch, fs, x);
mCombined = audio_metrics(yCombined, fs, x);
rows(end+1,:) = {'白噪声','低通',cfg.C.defaultLowpassOrder,sprintf('截止%.0f Hz',cfg.C.defaultLowpassCutoffHz),snrWhiteBefore,mLow.snr,mLow.rmse,'低通抑制部分高频白噪声'}; %#ok<AGROW>
rows(end+1,:) = {'单频干扰','陷波',2,sprintf('中心%.0f Hz',cfg.C.noiseFreqHz),snrToneBefore,mNotch.snr,mNotch.rmse,'陷波器针对3000 Hz窄带干扰'}; %#ok<AGROW>
rows(end+1,:) = {'混合噪声','低通+陷波',cfg.C.defaultLowpassOrder,sprintf('截止%.0f Hz，中心%.0f Hz',cfg.C.defaultLowpassCutoffHz,cfg.C.noiseFreqHz),snrMixedBefore,mCombined.snr,mCombined.rmse,'组合滤波兼顾宽带与单频噪声'}; %#ok<AGROW>

paramRows = {};
for i = 1:numel(cfg.C.lowpassOrders)
    % 比较不同低通滤波器阶数和截止频率对去噪效果的影响。
    order = cfg.C.lowpassOrders(i);
    cutoff = cfg.C.lowpassCutoffsHz(min(i, numel(cfg.C.lowpassCutoffsHz)));
    [b, a] = design_lowpass(order, cutoff, fs);
    y = normalize_audio_peak(apply_filter_safe(b, a, xMixed), 0.98);
    y = normalize_audio_peak(apply_filter_safe(bNotch, aNotch, y), 0.98);
    mp = audio_metrics(y, fs, x);
    paramRows(end+1,:) = {order, cutoff, mp.snr, mp.rmse}; %#ok<AGROW>
    rows(end+1,:) = {'混合噪声','参数对比：低通+陷波',order,sprintf('截止%.0f Hz',cutoff),snrMixedBefore,mp.snr,mp.rmse,'用于比较阶数和截止频率对去噪效果的影响'}; %#ok<AGROW>
end

T = cell2table(rows, 'VariableNames', {'噪声类型','滤波器类型','阶数','截止频率或中心频率','SNR_before','SNR_after','RMSE','说明'});
write_table_utf8(T, fullfile(tableDir, 'C_filter_metrics.csv'));
write_latex_table({'噪声类型','滤波器','阶数','参数','滤波前SNR','滤波后SNR','RMSE'}, ...
    rows(:, 1:7), fullfile(reportTableDir, 'C_filter_metrics_table.tex'));

plot_parameter_compare(paramRows, cfg, fullfile(figDir, 'C_filter_parameter_compare.png'));
plot_snr_rmse_compare(rows, cfg, fullfile(figDir, 'C_snr_rmse_compare.png'));

copy_outputs(figDir, reportFigDir, {'C_noise_waveform_spectrum_compare.png','C_filter_frequency_response.png', ...
    'C_denoise_waveform_compare.png','C_denoise_spectrum_compare.png', ...
    'C_filter_parameter_compare.png','C_snr_rmse_compare.png'});
fprintf('课题C完成：加噪、低通、陷波和组合滤波结果已生成。\n');
end

function ensure_dirs(dirs)
%ENSURE_DIRS 创建课题C所需的结果输出目录
for i = 1:numel(dirs)
    if ~exist(dirs{i}, 'dir'), mkdir(dirs{i}); end
end
end

function [b, a] = design_lowpass(order, cutoffHz, fs)
%DESIGN_LOWPASS 设计低通滤波器，优先使用Butterworth滤波器
wn = cutoffHz / (fs/2);
if exist('butter', 'file') == 2
    [b, a] = butter(order, wn, 'low');
else
    n = max(31, order*20+1);
    if mod(n,2)==0, n=n+1; end
    m = (-(n-1)/2:(n-1)/2)';
    h = 2*wn*sinc(2*wn*m);
    w = 0.54 - 0.46*cos(2*pi*(0:n-1)'/(n-1));
    b = (h.*w)';
    b = b / sum(b);
    a = 1;
end
end

function [b, a] = design_notch(f0, bw, fs)
%DESIGN_NOTCH 根据中心频率和带宽设计二阶陷波滤波器
w0 = 2*pi*f0/fs;
r = max(0.8, 1 - pi*bw/fs);
b = [1, -2*cos(w0), 1];
a = [1, -2*r*cos(w0), r^2];
end

function y = apply_filter_safe(b, a, x)
%APPLY_FILTER_SAFE 优先采用零相位滤波，必要时使用普通滤波
if exist('filtfilt', 'file') == 2 && numel(x) > 3*max(numel(a), numel(b))
    y = filtfilt(b, a, x);
else
    warning('未使用零相位filtfilt，普通filter可能引入相位延迟。');
    y = filter(b, a, x);
end
y = real(y(:));
end

function plot_noise_compare(x, xWhite, xTone, xMixed, fs, cfg, outputPath)
%PLOT_NOISE_COMPARE 绘制加噪前后的局部波形和频谱对比
fig = figure('Color','w','Visible',ternary(cfg.showFigures,'on','off'),'Position',[80,80,1200,840]);
t = (0:numel(x)-1)'/fs; idx = t <= 6;
tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
nexttile; plot(t(idx), x(idx),'k'); hold on; plot(t(idx), xWhite(idx),'b'); grid on; title('白噪声前后局部波形'); xlabel('时间 / 秒'); ylabel('幅值'); legend('原始','白噪声');
nexttile; plot(t(idx), x(idx),'k'); hold on; plot(t(idx), xTone(idx),'r'); grid on; title('单频干扰前后局部波形'); xlabel('时间 / 秒'); ylabel('幅值'); legend('原始','单频干扰');
nexttile; plot(t(idx), x(idx),'k'); hold on; plot(t(idx), xMixed(idx),'m'); grid on; title('混合噪声前后局部波形'); xlabel('时间 / 秒'); ylabel('幅值'); legend('原始','混合噪声');
nexttile; plot_spectrum_lines({x,xWhite,xTone,xMixed}, fs, {'原始','白噪声','单频干扰','混合噪声'}, cfg); title('加噪前后频谱对比');
save_figure_safely(fig, outputPath); if ~cfg.showFigures, close(fig); end
end

function plot_filter_response(bLp, aLp, bNotch, aNotch, fs, cfg, outputPath)
%PLOT_FILTER_RESPONSE 绘制低通滤波器和陷波滤波器的幅频响应
fig = figure('Color','w','Visible',ternary(cfg.showFigures,'on','off'),'Position',[80,80,1100,620]);
[h1, f1] = freq_response_local(bLp, aLp, fs);
[h2, f2] = freq_response_local(bNotch, aNotch, fs);
plot(f1, 20*log10(abs(h1)+eps),'b'); hold on; plot(f2, 20*log10(abs(h2)+eps),'r');
xlim([0 cfg.maxSpectrumFreqHz]); grid on; xlabel('频率 / Hz'); ylabel('幅度 / dB');
title('低通滤波器与陷波滤波器幅频响应'); legend('低通','陷波');
save_figure_safely(fig, outputPath); if ~cfg.showFigures, close(fig); end
end

function [h, f] = freq_response_local(b, a, fs)
%FREQ_RESPONSE_LOCAL 计算滤波器幅频响应
n = 4096;
if exist('freqz', 'file') == 2
    [h, f] = freqz(b, a, n, fs);
else
    impulse = [1; zeros(n-1,1)];
    resp = filter(b, a, impulse);
    H = fft(resp, 2*n);
    h = H(1:n);
    f = (0:n-1)' * fs / (2*n);
end
end

function plot_denoise_waveform(x, xMixed, yLow, yNotch, yCombined, fs, cfg, outputPath)
%PLOT_DENOISE_WAVEFORM 绘制混合噪声和滤波后信号的局部波形
fig = figure('Color','w','Visible',ternary(cfg.showFigures,'on','off'),'Position',[80,80,1200,760]);
t = (0:numel(x)-1)'/fs; idx = t <= 8;
plot(t(idx), x(idx),'k'); hold on; plot(t(idx), xMixed(idx),'Color',[0.7 0.3 0.7]); plot(t(idx), yLow(idx),'b'); plot(t(idx), yNotch(idx),'r'); plot(t(idx), yCombined(idx),'g');
grid on; xlabel('时间 / 秒'); ylabel('幅值'); title('混合噪声与滤波后局部波形对比');
legend('原始','混合噪声','低通','陷波','组合滤波');
save_figure_safely(fig, outputPath); if ~cfg.showFigures, close(fig); end
end

function plot_denoise_spectrum(x, xMixed, yLow, yNotch, yCombined, fs, cfg, outputPath)
%PLOT_DENOISE_SPECTRUM 绘制滤波前后的频谱对比
fig = figure('Color','w','Visible',ternary(cfg.showFigures,'on','off'),'Position',[80,80,1200,760]);
plot_spectrum_lines({x,xMixed,yLow,yNotch,yCombined}, fs, {'原始','混合噪声','低通','陷波','组合滤波'}, cfg);
title('滤波前后频谱对比');
save_figure_safely(fig, outputPath); if ~cfg.showFigures, close(fig); end
end

function plot_spectrum_lines(signals, fs, names, cfg)
%PLOT_SPECTRUM_LINES 在同一坐标轴绘制多组信号频谱
colors = lines(numel(signals));
hold on;
for i = 1:numel(signals)
    s = compute_fft_spectrum(signals{i}, fs);
    mask = s.freq <= cfg.maxSpectrumFreqHz;
    plot(s.freq(mask), s.db(mask), 'Color', colors(i,:));
end
grid on; xlabel('频率 / Hz'); ylabel('幅度 / dB'); legend(names, 'Location','best');
end

function plot_parameter_compare(paramRows, cfg, outputPath)
%PLOT_PARAMETER_COMPARE 绘制滤波参数对SNR的影响
fig = figure('Color','w','Visible',ternary(cfg.showFigures,'on','off'),'Position',[80,80,1100,620]);
orders = cell2mat(paramRows(:,1)); cutoffs = cell2mat(paramRows(:,2)); snrVals = cell2mat(paramRows(:,3));
bar(categorical(compose('%d阶/%dHz', orders, cutoffs)), snrVals);
grid on; ylabel('滤波后SNR / dB'); title('低通参数对组合去噪效果的影响');
save_figure_safely(fig, outputPath); if ~cfg.showFigures, close(fig); end
end

function plot_snr_rmse_compare(rows, cfg, outputPath)
%PLOT_SNR_RMSE_COMPARE 绘制不同滤波结果的SNR和RMSE指标
fig = figure('Color','w','Visible',ternary(cfg.showFigures,'on','off'),'Position',[80,80,1200,620]);
labels = strings(size(rows, 1), 1);
for i = 1:size(rows, 1)
    labels(i) = sprintf('%d-%s', i, string(rows{i, 2}));
end
snrAfter = cell2mat(rows(:,6));
rmseVal = cell2mat(rows(:,7));
tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');
nexttile; bar(categorical(labels), snrAfter); grid on; ylabel('SNR / dB'); title('不同滤波结果SNR');
nexttile; bar(categorical(labels), rmseVal); grid on; ylabel('RMSE'); title('不同滤波结果RMSE');
save_figure_safely(fig, outputPath); if ~cfg.showFigures, close(fig); end
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
