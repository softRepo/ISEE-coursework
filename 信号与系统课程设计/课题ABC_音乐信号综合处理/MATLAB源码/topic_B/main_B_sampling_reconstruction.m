function main_B_sampling_reconstruction(rootDir, cfg)
%课题B：音频采样、降采样与重构
%   读取 Hotel California 音频片段，估计最高有效频率。
%   比较过采样、临界采样和欠采样条件下的重构误差和混叠现象。

if nargin < 1 || isempty(rootDir)
    rootDir = fileparts(fileparts(mfilename('fullpath')));
end
if nargin < 2 || isempty(cfg)
    addpath(rootDir);
    cfg = config_ABC_experiments();
end
addpath(fullfile(rootDir, 'common'));

inPath = fullfile(rootDir, 'segments', cfg.segments.B);
figDir = fullfile(rootDir, 'results', 'B_sampling', 'figures');
audioDir = fullfile(rootDir, 'results', 'B_sampling', 'audio');
tableDir = fullfile(rootDir, 'results', 'B_sampling', 'tables');
reportFigDir = fullfile(rootDir, 'report', 'latex_report', 'figures');
reportTableDir = fullfile(rootDir, 'report', 'latex_report', 'tables');
ensure_dirs({figDir, audioDir, tableDir, reportFigDir, reportTableDir});

[x, fs] = load_audio_segment(inPath, cfg.targetFs);
[x, ~] = normalize_audio_peak(x, 0.98);
audiowrite(fullfile(audioDir, 'B_original.wav'), x, fs);

% 采用累计频谱能量估计工程意义上的最高有效频率。
bw = estimate_effective_bandwidth(x, fs, cfg.B.energyRatioForBandwidth, cfg.B.dbDropForBandwidth);
fEff = bw.fEffective;
fsOver = clamp_rate(cfg.B.oversamplingFactor * fEff, cfg.B.minSamplingRate, cfg.B.maxSamplingRate);
fsCritical = clamp_rate(2 * cfg.B.criticalFactor * fEff, cfg.B.minSamplingRate, cfg.B.maxSamplingRate);
fsUnder = clamp_rate(2 * cfg.B.undersamplingFactor * fEff, cfg.B.minSamplingRate, cfg.B.maxSamplingRate);
targets = [fsOver, fsCritical, fsUnder];
schemeNames = {'过采样','临界采样','欠采样'};

plot_bandwidth(x, fs, bw, cfg, fullfile(figDir, 'B_original_spectrum_bandwidth.png'));

rows = {};
reconStore = struct();
for i = 1:numel(targets)
    targetFs = targets(i);
    % 支路1：规范重采样，体现抗混叠滤波思想。
    [yd, fsActual] = resample_to_rate(x, fs, targetFs);
    yRecon = resample_to_rate(yd, fsActual, fs);
    yRecon = match_length(yRecon, numel(x));
    [yRecon, ~] = normalize_audio_peak(yRecon, 0.98);
    m = audio_metrics(yRecon, fs, x);
    rows(end+1, :) = {sprintf('%s-规范重采样', schemeNames{i}), targetFs, fsActual, '是', ...
        m.mse, m.rmse, m.corr, m.snr, '重采样支路含抗混叠滤波，重构失真较小'}; %#ok<AGROW>
    reconStore(i).anti = yRecon; %#ok<AGROW>
    audiowrite(fullfile(audioDir, sprintf('B_%s_resample_reconstructed.wav', lower_tag(schemeNames{i}))), yRecon, fs);

    % 支路2：直接抽取，不进行抗混叠滤波，用于观察混叠失真。
    M = max(1, round(fs / targetFs));
    ydDirect = x(1:M:end);
    fsDirect = fs / M;
    yDirectRecon = resample_to_rate(ydDirect, fsDirect, fs);
    yDirectRecon = match_length(yDirectRecon, numel(x));
    [yDirectRecon, ~] = normalize_audio_peak(yDirectRecon, 0.98);
    md = audio_metrics(yDirectRecon, fs, x);
    rows(end+1, :) = {sprintf('%s-直接抽取', schemeNames{i}), targetFs, fsDirect, '否', ...
        md.mse, md.rmse, md.corr, md.snr, sprintf('整数抽取因子M=%d，未做抗混叠滤波，可能产生混叠', M)}; %#ok<AGROW>
    reconStore(i).direct = yDirectRecon;
    audiowrite(fullfile(audioDir, sprintf('B_%s_direct_reconstructed.wav', lower_tag(schemeNames{i}))), yDirectRecon, fs);
end

T = cell2table(rows, 'VariableNames', {'采样方式','目标采样率','实际采样率','是否抗混叠','重构MSE','RMSE','相关系数','重构SNR','说明'});
write_table_utf8(T, fullfile(tableDir, 'B_sampling_metrics.csv'));
write_latex_table({'采样方式','实际采样率','抗混叠','RMSE','相关系数','SNR/dB'}, ...
    rows(:, [1 3 4 6 7 8]), fullfile(reportTableDir, 'B_sampling_metrics_table.tex'));

plot_sampling_time(x, reconStore, fs, cfg, schemeNames, fullfile(figDir, 'B_sampling_time_compare.png'));
plot_sampling_spectrum(x, reconStore, fs, cfg, schemeNames, fullfile(figDir, 'B_sampling_spectrum_compare.png'));
plot_aliasing(x, reconStore, fs, cfg, fullfile(figDir, 'B_aliasing_direct_decimation.png'));
plot_error_compare(x, reconStore, fs, cfg, schemeNames, fullfile(figDir, 'B_reconstruction_error_compare.png'));
plot_scheme_overview(targets, fEff, fs, schemeNames, cfg, fullfile(figDir, 'B_sampling_scheme_overview.png'));

copy_outputs(figDir, reportFigDir, {'B_original_spectrum_bandwidth.png','B_sampling_time_compare.png', ...
    'B_sampling_spectrum_compare.png','B_aliasing_direct_decimation.png', ...
    'B_reconstruction_error_compare.png','B_sampling_scheme_overview.png'});
fprintf('课题B完成：有效频率约 %.1f Hz，理论最低采样率约 %.1f Hz。\n', fEff, 2*fEff);
end

function ensure_dirs(dirs)
%ENSURE_DIRS 创建课题B所需的结果输出目录
for i = 1:numel(dirs)
    if ~exist(dirs{i}, 'dir'), mkdir(dirs{i}); end
end
end

function rate = clamp_rate(rate, minRate, maxRate)
%CLAMP_RATE 将采样率限制在合理范围内并取整到100 Hz
rate = max(minRate, min(maxRate, round(rate / 100) * 100));
end

function [y, fsOut] = resample_to_rate(x, fsIn, fsTarget)
%RESAMPLE_TO_RATE 将信号重采样到指定采样率
fsOut = round(fsTarget);
if abs(fsOut - fsIn) < 1e-9
    y = x;
    return;
end
if exist('resample', 'file') == 2
    [p, q] = rat(fsOut / fsIn, 1e-8);
    y = resample(x, p, q);
else
    tOld = (0:numel(x)-1)' / fsIn;
    nNew = max(1, round(numel(x) * fsOut / fsIn));
    tNew = (0:nNew-1)' / fsOut;
    y = interp1(tOld, x, tNew, 'linear', 'extrap');
end
end

function y = match_length(y, n)
%MATCH_LENGTH 将重构信号长度调整为参考信号长度
if numel(y) > n
    y = y(1:n);
elseif numel(y) < n
    y = [y(:); zeros(n-numel(y), 1)];
end
end

function plot_bandwidth(x, fs, bw, cfg, outputPath)
%PLOT_BANDWIDTH 绘制频谱和累计能量曲线以展示有效带宽估计
fig = figure('Color','w','Visible',ternary(cfg.showFigures,'on','off'),'Position',[80,80,1200,760]);
tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');
mask = bw.freq <= cfg.maxSpectrumFreqHz;
nexttile; plot(bw.freq(mask), bw.db(mask),'b'); hold on; xline(bw.fEffective,'r--','有效最高频率');
grid on; xlabel('频率 / Hz'); ylabel('幅度 / dB'); title('原始片段dB频谱与有效带宽估计');
nexttile; plot(bw.freq(mask), bw.cumulativeEnergy(mask),'k'); hold on; yline(bw.energyRatio,'r--','能量比例阈值');
grid on; xlabel('频率 / Hz'); ylabel('累计能量比例'); title('累计频谱能量曲线');
save_figure_safely(fig, outputPath); if ~cfg.showFigures, close(fig); end
end

function plot_sampling_time(x, reconStore, fs, cfg, names, outputPath)
%PLOT_SAMPLING_TIME 绘制不同采样方式的局部时域重构对比
fig = figure('Color','w','Visible',ternary(cfg.showFigures,'on','off'),'Position',[80,80,1200,760]);
t = (0:numel(x)-1)'/fs; idx = t>=cfg.B.displayTimeRangeSec(1) & t<=cfg.B.displayTimeRangeSec(2);
tiledlayout(fig,3,1,'TileSpacing','compact','Padding','compact');
for i = 1:3
    nexttile; plot(t(idx), x(idx),'k'); hold on; plot(t(idx), reconStore(i).anti(idx),'b'); plot(t(idx), reconStore(i).direct(idx),'r');
    grid on; xlabel('时间 / 秒'); ylabel('幅值'); title([names{i}, '重构局部波形']); legend('原始','规范重采样','直接抽取');
end
save_figure_safely(fig, outputPath); if ~cfg.showFigures, close(fig); end
end

function plot_sampling_spectrum(x, reconStore, fs, cfg, names, outputPath)
%PLOT_SAMPLING_SPECTRUM 绘制不同采样方式的重构频谱对比
fig = figure('Color','w','Visible',ternary(cfg.showFigures,'on','off'),'Position',[80,80,1200,760]);
tiledlayout(fig,3,1,'TileSpacing','compact','Padding','compact');
s0 = compute_fft_spectrum(x, fs); mask = s0.freq <= cfg.maxSpectrumFreqHz;
for i = 1:3
    nexttile; plot(s0.freq(mask), s0.db(mask),'k'); hold on;
    s1 = compute_fft_spectrum(reconStore(i).anti, fs); s2 = compute_fft_spectrum(reconStore(i).direct, fs);
    plot(s1.freq(mask), s1.db(mask),'b'); plot(s2.freq(mask), s2.db(mask),'r');
    grid on; xlabel('频率 / Hz'); ylabel('幅度 / dB'); title([names{i}, '重构频谱对比']); legend('原始','规范重采样','直接抽取');
end
save_figure_safely(fig, outputPath); if ~cfg.showFigures, close(fig); end
end

function plot_aliasing(x, reconStore, fs, cfg, outputPath)
%PLOT_ALIASING 绘制欠采样直接抽取引起的混叠频谱
fig = figure('Color','w','Visible',ternary(cfg.showFigures,'on','off'),'Position',[80,80,1200,620]);
s0 = compute_fft_spectrum(x, fs); sd = compute_fft_spectrum(reconStore(3).direct, fs);
mask = s0.freq <= cfg.maxSpectrumFreqHz;
plot(s0.freq(mask), s0.db(mask),'k'); hold on; plot(sd.freq(mask), sd.db(mask),'r');
grid on; xlabel('频率 / Hz'); ylabel('幅度 / dB'); title('欠采样直接抽取引起的混叠频谱对比'); legend('原始','欠采样直接抽取重构');
save_figure_safely(fig, outputPath); if ~cfg.showFigures, close(fig); end
end

function plot_error_compare(x, reconStore, fs, cfg, names, outputPath)
%PLOT_ERROR_COMPARE 绘制不同采样方式的时域重构误差
fig = figure('Color','w','Visible',ternary(cfg.showFigures,'on','off'),'Position',[80,80,1200,760]);
t = (0:numel(x)-1)'/fs; idx = t>=cfg.B.displayTimeRangeSec(1) & t<=cfg.B.displayTimeRangeSec(2);
tiledlayout(fig,3,1,'TileSpacing','compact','Padding','compact');
for i = 1:3
    nexttile; plot(t(idx), x(idx)-reconStore(i).anti(idx),'b'); hold on; plot(t(idx), x(idx)-reconStore(i).direct(idx),'r');
    grid on; xlabel('时间 / 秒'); ylabel('误差'); title([names{i}, '重构误差']); legend('规范重采样','直接抽取');
end
save_figure_safely(fig, outputPath); if ~cfg.showFigures, close(fig); end
end

function plot_scheme_overview(targets, fEff, fs, names, cfg, outputPath)
%PLOT_SCHEME_OVERVIEW 绘制过采样、临界采样和欠采样方案
fig = figure('Color','w','Visible',ternary(cfg.showFigures,'on','off'),'Position',[80,80,1000,620]);
bar(categorical(names), targets); hold on; yline(2*fEff,'r--','理论最低采样率'); yline(fs,'k:','原始采样率');
grid on; ylabel('采样率 / Hz'); title('过采样、临界采样与欠采样方案');
save_figure_safely(fig, outputPath); if ~cfg.showFigures, close(fig); end
end

function copy_outputs(srcDir, dstDir, names)
%COPY_OUTPUTS 将指定图像复制到配套材料目录
for i = 1:numel(names)
    copyfile(fullfile(srcDir, names{i}), fullfile(dstDir, names{i}));
end
end

function tag = lower_tag(name)
%LOWER_TAG 将中文采样方式转换为音频文件名标签
switch name
    case '过采样', tag = 'over';
    case '临界采样', tag = 'critical';
    otherwise, tag = 'under';
end
end

function y = ternary(cond, a, b)
%TERNARY 根据逻辑条件返回两个候选值之一
if cond, y = a; else, y = b; end
end
