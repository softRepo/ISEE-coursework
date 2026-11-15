%% 课题E：基于 MFCC 特征与 DTW 算法的孤立数字语音识别系统
% 本脚本为课题E孤立数字语音识别实验的主入口。
% 运行时应从 matlab_topicE 目录进入，并执行：
% main_topicE_dtw_speech_recognition

clear; clc; close all;

projectDir = fileparts(mfilename('fullpath'));
if isempty(projectDir)
    projectDir = pwd;
end
addpath(fullfile(projectDir, 'functions'));

cfg = config_topicE();
ensure_output_dirs(cfg);

diaryFile = fullfile(cfg.logsDir, 'run_log.txt');
if exist(diaryFile, 'file')
    delete(diaryFile);
end
diary(diaryFile);

try
    fprintf('课题E：基于 MFCC 与 DTW 的孤立数字语音识别系统\n');
    fprintf('运行时间：%s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf('工程目录：%s\n', cfg.projectDir);
    fprintf('数据目录：%s\n\n', cfg.recordingsDir);

    rng(cfg.randomSeed);
    totalTimer = tic;

    %% 1. 扫描数据集并按录音编号划分训练集与测试集
    dataTbl = scan_fsdd_dataset(cfg.recordingsDir);
    dataTbl = dataTbl(ismember(dataTbl.Digit, cfg.digits), :);

    dataTbl.SetName = repmat({'未使用'}, height(dataTbl), 1);
    trainMask = ismember(dataTbl.Index, cfg.trainIndexList);
    testMask = ismember(dataTbl.Index, cfg.testIndexList);
    dataTbl.SetName(trainMask) = repmat({'训练模板候选'}, sum(trainMask), 1);
    dataTbl.SetName(testMask) = repmat({'测试'}, sum(testMask), 1);

    trainPoolTbl = dataTbl(trainMask, :);
    testTbl = dataTbl(testMask, :);
    testTbl = limit_tests_per_digit(testTbl, cfg);
    templateTbl = select_templates_balanced(trainPoolTbl, cfg);

    fprintf('扫描到 wav 文件：%d 条\n', height(dataTbl));
    fprintf('训练候选样本：%d 条\n', height(trainPoolTbl));
    fprintf('实际模板样本：%d 条\n', height(templateTbl));
    fprintf('测试样本：%d 条\n\n', height(testTbl));

    plot_dataset_statistics(dataTbl, cfg);

    %% 2. 提取训练模板的 MFCC 特征
    nTemplates = height(templateTbl);
    templates = repmat(make_empty_template_struct(), nTemplates, 1);

    fprintf('开始提取训练模板 MFCC 特征...\n');
    for k = 1:nTemplates
        [x, fs] = load_audio_mono(templateTbl.FilePath{k});
        [speech, prepInfo] = preprocess_speech(x, fs, cfg);
        [features, mfccInfo] = compute_mfcc_manual(speech, cfg.targetFs, cfg);

        templates(k).digit = templateTbl.Digit(k);
        templates(k).speaker = templateTbl.Speaker{k};
        templates(k).index = templateTbl.Index(k);
        templates(k).fileName = templateTbl.FileName{k};
        templates(k).filePath = templateTbl.FilePath{k};
        templates(k).features = features;
        templates(k).prepInfo = prepInfo;
        templates(k).mfccInfo = mfccInfo;

        if mod(k, 30) == 0 || k == nTemplates
            fprintf('  模板特征进度：%d / %d\n', k, nTemplates);
        end
    end

    %% 3. 对测试样本逐条进行 DTW 模板匹配分类
    nTests = height(testTbl);
    digits = cfg.digits(:).';
    numClasses = numel(digits);

    fileNameCol = cell(nTests, 1);
    speakerCol = cell(nTests, 1);
    trueDigitCol = zeros(nTests, 1);
    predDigitCol = zeros(nTests, 1);
    minDistCol = zeros(nTests, 1);
    correctCol = false(nTests, 1);
    classDistMat = inf(nTests, numClasses);
    testFeatures = cell(nTests, 1);
    testPrepInfo = cell(nTests, 1);
    testMfccInfo = cell(nTests, 1);

    example = struct('hasData', false);

    fprintf('\n开始测试集识别...\n');
    for i = 1:nTests
        [x, fs] = load_audio_mono(testTbl.FilePath{i});
        [speech, prepInfo] = preprocess_speech(x, fs, cfg);
        [features, mfccInfo] = compute_mfcc_manual(speech, cfg.targetFs, cfg);
        [predDigit, classDistances, detail] = classify_by_dtw(features, templates, cfg);

        fileNameCol{i} = testTbl.FileName{i};
        speakerCol{i} = testTbl.Speaker{i};
        trueDigitCol(i) = testTbl.Digit(i);
        predDigitCol(i) = predDigit;
        classDistMat(i, :) = classDistances;
        minDistCol(i) = min(classDistances);
        correctCol(i) = predDigit == testTbl.Digit(i);
        testFeatures{i} = features;
        testPrepInfo{i} = prepInfo;
        testMfccInfo{i} = mfccInfo;

        fprintf('%3d/%3d  文件：%-22s  真实：%d  预测：%d  最小DTW距离：%.4f  %s\n', ...
            i, nTests, testTbl.FileName{i}, trueDigitCol(i), predDigitCol(i), ...
            minDistCol(i), ternary(correctCol(i), '正确', '错误'));

        if ~example.hasData && i >= cfg.exampleTestOrder
            example.hasData = true;
            example.testIndex = i;
            example.rawSignal = x;
            example.rawFs = fs;
            example.speech = speech;
            example.prepInfo = prepInfo;
            example.features = features;
            example.mfccInfo = mfccInfo;
            example.classDistances = classDistances;
            example.detail = detail;
            example.fileName = testTbl.FileName{i};
            example.trueDigit = testTbl.Digit(i);
            example.predDigit = predDigit;
        end
    end

    %% 4. 统计识别准确率并保存结果表格
    [confMat, classAccTbl, overallAccuracy] = evaluate_predictions(trueDigitCol, predDigitCol, digits);
    predictionTbl = make_prediction_table(fileNameCol, speakerCol, trueDigitCol, predDigitCol, ...
        minDistCol, correctCol, classDistMat, digits);

    writetable(predictionTbl, fullfile(cfg.tablesDir, 'prediction_results.csv'));
    writetable(classAccTbl, fullfile(cfg.tablesDir, 'class_accuracy.csv'));
    confCsvTbl = array2table([digits(:) confMat], ...
        'VariableNames', [{'TrueDigit'}, make_pred_var_names(digits)]);
    writetable(confCsvTbl, fullfile(cfg.tablesDir, 'confusion_matrix.csv'));

    fprintf('\n总体准确率：%.2f%% (%d / %d)\n', overallAccuracy, sum(correctCol), nTests);
    for c = 1:height(classAccTbl)
        fprintf('数字 %d：%.2f%% (%d / %d)\n', classAccTbl.Digit(c), ...
            classAccTbl.AccuracyPercent(c), classAccTbl.Correct(c), classAccTbl.Total(c));
    end

    %% 5. 绘制并保存典型实验图像
    if example.hasData
        plot_preprocess_example(example.rawSignal, example.rawFs, example.prepInfo, ...
            fullfile(cfg.figuresDir, 'preprocess_example.png'), cfg, example.fileName);
        plot_mfcc_heatmap(example.features, fullfile(cfg.figuresDir, 'mfcc_heatmap_example.png'), cfg, example.fileName);
        plot_class_distance_bar(digits, example.classDistances, ...
            fullfile(cfg.figuresDir, 'class_distance_bar_example.png'), cfg, example.fileName);

        bestTemplateIndex = example.detail.bestTemplateIndex;
        if isfinite(bestTemplateIndex) && bestTemplateIndex >= 1 && bestTemplateIndex <= numel(templates)
            [~, dtwInfo] = dtw_distance_manual(example.features, templates(bestTemplateIndex).features, cfg, true);
            plot_dtw_example(example.features, templates(bestTemplateIndex).features, dtwInfo, ...
                cfg.figuresDir, 'dtw_example', cfg);
        end
    end

    plot_confusion_matrix_local(confMat, digits, fullfile(cfg.figuresDir, 'confusion_matrix.png'), cfg);
    plot_class_accuracy(classAccTbl, fullfile(cfg.figuresDir, 'class_accuracy_bar.png'), cfg);

    %% 6. 保存特征缓存、运行记录和辅助输出文件
    runtimeSec = toc(totalTimer);
    if cfg.saveFeatureMat
        save(fullfile(cfg.featuresDir, 'feature_cache.mat'), ...
            'templates', 'testFeatures', 'testPrepInfo', 'testMfccInfo', ...
            'templateTbl', 'testTbl', 'cfg', '-v7.3');
    end

    stats = struct();
    stats.numAllSamples = height(dataTbl);
    stats.numTrainCandidates = height(trainPoolTbl);
    stats.numTemplates = nTemplates;
    stats.numTests = nTests;
    stats.numCorrect = sum(correctCol);
    stats.overallAccuracy = overallAccuracy;
    stats.runtimeSec = runtimeSec;
    stats.confMat = confMat;
    stats.classAccTbl = classAccTbl;
    stats.predictionTbl = predictionTbl;

    write_summary_log(cfg, stats);
    generate_markdown_reports(cfg, stats);
    generate_latex_result_files(cfg, stats);
    copy_report_assets(cfg);

    fprintf('\n全部流程完成，用时 %.2f 秒。\n', runtimeSec);
    fprintf('结果目录：%s\n', cfg.resultsDir);
    fprintf('报告目录：%s\n', cfg.reportDir);

catch ME
    fprintf(2, '\n运行失败：%s\n', ME.message);
    for k = 1:numel(ME.stack)
        fprintf(2, '  %s 第 %d 行\n', ME.stack(k).file, ME.stack(k).line);
    end
    diary off;
    rethrow(ME);
end

diary off;

%% ===== 主脚本局部辅助函数 =====

function ensure_output_dirs(cfg)
% 创建实验输出所需目录，保证图像、表格和运行记录可以正常保存。
dirs = {cfg.resultsDir, cfg.figuresDir, cfg.tablesDir, cfg.featuresDir, cfg.logsDir, ...
    cfg.reportDir, cfg.latexDir, cfg.latexFiguresDir, cfg.latexTablesDir, cfg.latexCodeDir};
for k = 1:numel(dirs)
    if ~exist(dirs{k}, 'dir')
        mkdir(dirs{k});
    end
end
end

function tbl = limit_tests_per_digit(testTbl, cfg)
% 按数字类别限制测试样本数量，便于在不同规模下复现实验。
if isinf(cfg.maxTestsPerDigit)
    tbl = sortrows(testTbl, {'Digit', 'Speaker', 'Index'});
    return;
end

selected = [];
for d = cfg.digits
    rows = find(testTbl.Digit == d);
    [~, ord] = sortrows([cellstr_to_rank(testTbl.Speaker(rows)), testTbl.Index(rows)]);
    rows = rows(ord);
    selected = [selected; rows(1:min(cfg.maxTestsPerDigit, numel(rows)))]; %#ok<AGROW>
end
tbl = sortrows(testTbl(selected, :), {'Digit', 'Speaker', 'Index'});
end

function templateTbl = select_templates_balanced(trainPoolTbl, cfg)
% 按数字类别选择训练模板，并尽量保持不同说话人之间的样本均衡。
selected = [];
for d = cfg.digits
    digitRows = find(trainPoolTbl.Digit == d);
    if isempty(digitRows)
        continue;
    end
    if isinf(cfg.maxTemplatesPerDigit)
        selected = [selected; digitRows]; %#ok<AGROW>
        continue;
    end

    maxCount = min(cfg.maxTemplatesPerDigit, numel(digitRows));
    if strcmpi(cfg.templateSelectionMode, 'balanced_by_speaker')
        speakers = unique(trainPoolTbl.Speaker(digitRows), 'stable');
        numSpeakers = numel(speakers);
        baseQuota = floor(maxCount / numSpeakers);
        remain = maxCount - baseQuota * numSpeakers;
        chosen = [];

        for s = 1:numSpeakers
            rows = digitRows(strcmp(trainPoolTbl.Speaker(digitRows), speakers{s}));
            [~, order] = sort(trainPoolTbl.Index(rows));
            rows = rows(order);
            quota = baseQuota + double(s <= remain);
            chosen = [chosen; rows(1:min(quota, numel(rows)))]; %#ok<AGROW>
        end

        if numel(chosen) < maxCount
            rest = setdiff(digitRows, chosen, 'stable');
            [~, order] = sort(trainPoolTbl.Index(rest));
            rest = rest(order);
            chosen = [chosen; rest(1:min(maxCount-numel(chosen), numel(rest)))]; %#ok<AGROW>
        end
    else
        [~, order] = sort(trainPoolTbl.Index(digitRows));
        chosen = digitRows(order(1:maxCount));
    end
    selected = [selected; chosen(:)]; %#ok<AGROW>
end

templateTbl = sortrows(trainPoolTbl(selected, :), {'Digit', 'Speaker', 'Index'});
end

function rank = cellstr_to_rank(names)
% 将说话人名称转换为稳定排序编号，用于样本排序。
uniqueNames = unique(names, 'stable');
rank = zeros(numel(names), 1);
for k = 1:numel(uniqueNames)
    rank(strcmp(names, uniqueNames{k})) = k;
end
end

function s = make_empty_template_struct()
% 构造模板样本结构体，统一保存标签、文件信息和特征数据。
s = struct('digit', [], 'speaker', '', 'index', [], 'fileName', '', ...
    'filePath', '', 'features', [], 'prepInfo', [], 'mfccInfo', []);
end

function [confMat, classAccTbl, overallAccuracy] = evaluate_predictions(trueDigit, predDigit, digits)
% 根据真实标签和预测标签计算混淆矩阵、每类准确率和总体准确率。
numClasses = numel(digits);
confMat = zeros(numClasses, numClasses);
for i = 1:numel(trueDigit)
    r = find(digits == trueDigit(i), 1);
    c = find(digits == predDigit(i), 1);
    if ~isempty(r) && ~isempty(c)
        confMat(r, c) = confMat(r, c) + 1;
    end
end

total = sum(confMat, 2);
correct = diag(confMat);
acc = 100 * correct ./ max(total, 1);
classAccTbl = table(digits(:), total, correct, acc, ...
    'VariableNames', {'Digit', 'Total', 'Correct', 'AccuracyPercent'});
overallAccuracy = 100 * sum(correct) / max(sum(total), 1);
end

function predictionTbl = make_prediction_table(fileName, speaker, trueDigit, predDigit, ...
    minDist, correct, classDistMat, digits)
% 汇总逐条测试样本的预测结果和各类别 DTW 距离，形成 CSV 输出表。
predictionTbl = table(fileName, speaker, trueDigit, predDigit, minDist, correct, ...
    'VariableNames', {'FileName', 'Speaker', 'TrueDigit', 'PredictedDigit', 'MinDistance', 'Correct'});
for c = 1:numel(digits)
    predictionTbl.(sprintf('Dist_%d', digits(c))) = classDistMat(:, c);
end
end

function names = make_pred_var_names(digits)
% 生成混淆矩阵中预测类别列名。
names = cell(1, numel(digits));
for k = 1:numel(digits)
    names{k} = sprintf('Pred_%d', digits(k));
end
end

function plot_dataset_statistics(dataTbl, cfg)
% 绘制数据集中各数字类别的样本数量统计图。
digits = cfg.digits(:).';
counts = zeros(size(digits));
for k = 1:numel(digits)
    counts(k) = sum(dataTbl.Digit == digits(k));
end

fig = figure('Visible', cfg.figureVisible, 'Color', 'w', 'Name', '数据集类别数量统计');
bar(digits, counts, 0.65, 'FaceColor', [0.20 0.48 0.72]);
grid on;
xlabel('数字类别');
ylabel('样本数量');
title('FSDD 数据集 0-9 数字类别数量统计');
set(gca, 'XTick', digits, 'FontName', 'Microsoft YaHei');
for k = 1:numel(digits)
    text(digits(k), counts(k), sprintf('%d', counts(k)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end
save_figure_safely(fig, fullfile(cfg.figuresDir, 'dataset_digit_counts.png'), cfg);
end

function plot_preprocess_example(rawSignal, rawFs, prepInfo, filePath, cfg, fileName)
% 绘制语音预处理示例，包括波形、短时能量、过零率和端点区域。
sig = prepInfo.preEmphasizedSignal;
fs = cfg.targetFs;
t = (0:numel(sig)-1) / fs;
startSec = (prepInfo.endpoint.startSample - 1) / fs;
endSec = (prepInfo.endpoint.endSample - 1) / fs;

fig = figure('Visible', cfg.figureVisible, 'Color', 'w', 'Name', '语音预处理示例');

subplot(3, 1, 1);
plot(t, sig, 'Color', [0.12 0.30 0.55], 'LineWidth', 0.8);
hold on;
yl = ylim;
patch([startSec endSec endSec startSec], [yl(1) yl(1) yl(2) yl(2)], ...
    [1.00 0.84 0.45], 'FaceAlpha', 0.28, 'EdgeColor', 'none');
plot(t, sig, 'Color', [0.12 0.30 0.55], 'LineWidth', 0.8);
hold off;
grid on;
xlabel('时间 / s');
ylabel('幅值');
title(sprintf('原始语音去直流、归一化与预加重后波形：%s', fileName), 'Interpreter', 'none');
legend({'有效语音区域', '波形'}, 'Location', 'best');

subplot(3, 1, 2);
plot(prepInfo.endpoint.frameTimeSec, prepInfo.endpoint.energyNorm, 'LineWidth', 1.1);
hold on;
yline(prepInfo.endpoint.energyThreshold, '--r', '能量阈值');
hold off;
grid on;
xlabel('时间 / s');
ylabel('归一化短时能量');
title('短时能量及自适应阈值');

subplot(3, 1, 3);
plot(prepInfo.endpoint.frameTimeSec, prepInfo.endpoint.zcrNorm, 'LineWidth', 1.1, 'Color', [0.55 0.20 0.55]);
hold on;
yline(prepInfo.endpoint.zcrThreshold, '--r', '过零率阈值');
hold off;
grid on;
xlabel('时间 / s');
ylabel('归一化过零率');
title('短时过零率及自适应阈值');

set(findall(fig, '-property', 'FontName'), 'FontName', 'Microsoft YaHei');
save_figure_safely(fig, filePath, cfg);

% 保留 rawSignal 和 rawFs 输入，便于扩展原始采样率波形对比。
unused = rawSignal; %#ok<NASGU>
unusedFs = rawFs; %#ok<NASGU>
end

function plot_mfcc_heatmap(features, filePath, cfg, fileName)
% 绘制 MFCC 特征热力图，用于观察特征随时间帧的变化。
fig = figure('Visible', cfg.figureVisible, 'Color', 'w', 'Name', 'MFCC特征热力图');
imagesc(features.');
axis xy;
colormap(turbo_safe());
colorbar;
xlabel('帧序号');
ylabel('MFCC 特征维度');
title(sprintf('MFCC 特征热力图：%s', fileName), 'Interpreter', 'none');
set(gca, 'FontName', 'Microsoft YaHei');
save_figure_safely(fig, filePath, cfg);
end

function cmap = turbo_safe()
% 根据 MATLAB 版本选择可用的颜色映射。
if exist('turbo', 'file') || exist('turbo', 'builtin')
    cmap = turbo;
else
    cmap = parula;
end
end

function plot_class_distance_bar(digits, classDistances, filePath, cfg, fileName)
% 绘制测试样本到各数字类别的 DTW 距离柱状图。
plotDistances = classDistances;
finiteMask = isfinite(plotDistances);
if any(finiteMask)
    capValue = max(plotDistances(finiteMask)) * 1.15;
else
    capValue = 1;
end
plotDistances(~finiteMask) = capValue;

fig = figure('Visible', cfg.figureVisible, 'Color', 'w', 'Name', '类别DTW距离柱状图');
bar(digits, plotDistances, 0.65, 'FaceColor', [0.26 0.58 0.50]);
grid on;
xlabel('候选数字类别');
ylabel('类别 DTW 距离');
title(sprintf('测试样本到 10 个数字类别的 DTW 距离：%s', fileName), 'Interpreter', 'none');
set(gca, 'XTick', digits, 'FontName', 'Microsoft YaHei');
save_figure_safely(fig, filePath, cfg);
end

function plot_class_accuracy(classAccTbl, filePath, cfg)
% 绘制每个数字类别的识别准确率柱状图。
fig = figure('Visible', cfg.figureVisible, 'Color', 'w', 'Name', '每类识别准确率');
bar(classAccTbl.Digit, classAccTbl.AccuracyPercent, 0.65, 'FaceColor', [0.74 0.35 0.22]);
grid on;
ylim([0 100]);
xlabel('数字类别');
ylabel('准确率 / %');
title('每个数字类别的识别准确率');
set(gca, 'XTick', classAccTbl.Digit, 'FontName', 'Microsoft YaHei');
for k = 1:height(classAccTbl)
    text(classAccTbl.Digit(k), classAccTbl.AccuracyPercent(k), ...
        sprintf('%.1f%%', classAccTbl.AccuracyPercent(k)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end
save_figure_safely(fig, filePath, cfg);
end

function write_summary_log(cfg, stats)
% 将实验参数、样本数量和识别准确率写入摘要文件。
summaryPath = fullfile(cfg.logsDir, 'experiment_summary.txt');
fid = fopen(summaryPath, 'w', 'n', 'UTF-8');
if fid < 0
    warning('无法写入 summary 文件：%s', summaryPath);
    return;
end
cleanup = onCleanup(@() fclose(fid));

fprintf(fid, '课题E：基于 MFCC 特征与 DTW 算法的孤立数字语音识别系统\n');
fprintf(fid, '生成时间：%s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, '数据集路径：%s\n', cfg.recordingsDir);
fprintf(fid, '识别数字范围：%s\n', mat2str(cfg.digits));
fprintf(fid, '训练编号：%s\n', mat2str(cfg.trainIndexList));
fprintf(fid, '测试编号：%s\n', mat2str(cfg.testIndexList));
fprintf(fid, '训练候选数量：%d\n', stats.numTrainCandidates);
fprintf(fid, '训练模板数量：%d\n', stats.numTemplates);
fprintf(fid, '测试样本数量：%d\n', stats.numTests);
fprintf(fid, '采样率：%d Hz\n\n', cfg.targetFs);

fprintf(fid, 'MFCC 参数：帧长 %.1f ms，帧移 %.1f ms，NFFT=%d，Mel滤波器=%d，MFCC维数=%d，使用C0=%d，使用Delta=%d，使用DeltaDelta=%d\n', ...
    cfg.frameLengthMs, cfg.frameShiftMs, cfg.nfft, cfg.numMelFilters, cfg.numMfcc, ...
    cfg.useC0, cfg.useDelta, cfg.useDeltaDelta);
fprintf(fid, 'DTW 参数：距离=%s，类别决策=%s，Sakoe-Chiba=%d，带宽比例=%.2f\n\n', ...
    cfg.dtwDistanceMetric, cfg.classDecisionMode, cfg.useSakoeChibaBand, cfg.sakoeChibaRatio);

fprintf(fid, '总体准确率：%.2f%% (%d / %d)\n', stats.overallAccuracy, stats.numCorrect, stats.numTests);
fprintf(fid, '每类准确率：\n');
for k = 1:height(stats.classAccTbl)
    fprintf(fid, '  数字 %d：%.2f%% (%d / %d)\n', stats.classAccTbl.Digit(k), ...
        stats.classAccTbl.AccuracyPercent(k), stats.classAccTbl.Correct(k), stats.classAccTbl.Total(k));
end
fprintf(fid, '\n运行时间：%.2f 秒\n', stats.runtimeSec);
fprintf(fid, '结果保存路径：%s\n', cfg.resultsDir);
fprintf(fid, '报告素材路径：%s\n', cfg.reportDir);
end

function generate_markdown_reports(cfg, stats)
% 根据本次实验统计结果写出便于查看的图表说明文本。
draftPath = fullfile(cfg.reportDir, 'report_topicE_draft.md');
fid = fopen(draftPath, 'w', 'n', 'UTF-8');
if fid >= 0
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# 课题E 简单语音信号识别报告草稿\n\n');
    fprintf(fid, '## 实验题目\n\n基于 MFCC 特征与 DTW 算法的孤立数字语音识别系统。\n\n');
    fprintf(fid, '## 数据与划分\n\n');
    fprintf(fid, '- 数据集：Free Spoken Digit Dataset，路径 `%s`\n', cfg.recordingsDir);
    fprintf(fid, '- 识别类别：数字 0 到 9，共 10 类。\n');
    fprintf(fid, '- 测试集：录音编号 0-4；训练模板候选：录音编号 5-49。\n');
    fprintf(fid, '- 实际训练模板数：%d；测试样本数：%d。\n\n', stats.numTemplates, stats.numTests);
    fprintf(fid, '## 方法概述\n\n');
    fprintf(fid, '系统先对语音进行去直流、归一化、可选预加重、短时能量与过零率端点检测；随后手工提取 MFCC 特征；最后使用手写 DTW 计算测试样本与模板之间的时间规整距离，并按类别汇总距离完成分类。\n\n');
    fprintf(fid, '## 实验结果\n\n');
    fprintf(fid, '- 总体准确率：%.2f%% (%d/%d)\n', stats.overallAccuracy, stats.numCorrect, stats.numTests);
    for k = 1:height(stats.classAccTbl)
        fprintf(fid, '- 数字 %d：%.2f%% (%d/%d)\n', stats.classAccTbl.Digit(k), ...
            stats.classAccTbl.AccuracyPercent(k), stats.classAccTbl.Correct(k), stats.classAccTbl.Total(k));
    end
    fprintf(fid, '\n## 结果文件\n\n');
    fprintf(fid, '- `results/tables/prediction_results.csv`：逐条测试样本预测结果。\n');
    fprintf(fid, '- `results/tables/class_accuracy.csv`：每类准确率。\n');
    fprintf(fid, '- `results/tables/confusion_matrix.csv`：十分类混淆矩阵。\n');
    fprintf(fid, '- `results/logs/experiment_summary.txt`：实验摘要日志。\n');
end

notesPath = fullfile(cfg.reportDir, 'experiment_notes.md');
fid = fopen(notesPath, 'w', 'n', 'UTF-8');
if fid >= 0
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '# 图表与表格说明\n\n');
    fprintf(fid, '- `dataset_digit_counts.png`：说明 FSDD 中 0-9 类样本数量均衡，每类 300 条。\n');
    fprintf(fid, '- `preprocess_example.png`：展示波形、短时能量、过零率及端点检测区域，用于说明预处理流程。\n');
    fprintf(fid, '- `mfcc_heatmap_example.png`：展示 MFCC 随时间变化的谱包络特征。\n');
    fprintf(fid, '- `class_distance_bar_example.png`：展示单个测试样本到 10 个类别的 DTW 距离，最低距离对应预测类别。\n');
    fprintf(fid, '- `dtw_example_local_distance.png`：展示测试帧与模板帧两两之间的局部距离。\n');
    fprintf(fid, '- `dtw_example_accumulated_path.png`：展示累积距离矩阵及最优路径，说明 DTW 如何处理时间伸缩。\n');
    fprintf(fid, '- `confusion_matrix.png`：展示真实类别和预测类别之间的对应关系，用于分析易混淆数字。\n');
    fprintf(fid, '- `class_accuracy_bar.png`：展示每个数字类别的识别准确率。\n');
    fprintf(fid, '- `prediction_results.csv`：报告中可引用典型正确和错误样本。\n');
    fprintf(fid, '- `class_accuracy.csv` 与 `confusion_matrix.csv`：用于报告第 5 节实验结果统计。\n');
end
end

function generate_latex_result_files(cfg, stats)
% 将关键实验结果写成表格源文件，便于后续整理实验材料。
write_latex_summary_table(fullfile(cfg.latexTablesDir, 'result_summary.tex'), cfg, stats);
write_latex_class_accuracy_table(fullfile(cfg.latexTablesDir, 'class_accuracy_table.tex'), stats.classAccTbl);
write_latex_confusion_table(fullfile(cfg.latexTablesDir, 'confusion_matrix_table.tex'), stats.confMat, cfg.digits);
end

function write_latex_summary_table(filePath, cfg, stats)
% 写出实验关键参数与总体结果表。
fid = fopen(filePath, 'w', 'n', 'UTF-8');
if fid < 0
    return;
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '\\begin{table}[H]\n\\centering\n');
fprintf(fid, '\\caption{实验关键参数与总体结果}\n');
fprintf(fid, '\\begin{tabular}{ll}\n\\toprule\n项目 & 数值 \\\\\n\\midrule\n');
fprintf(fid, '训练模板数量 & %d \\\\\n', stats.numTemplates);
fprintf(fid, '测试样本数量 & %d \\\\\n', stats.numTests);
fprintf(fid, '采样率 & %d Hz \\\\\n', cfg.targetFs);
fprintf(fid, 'MFCC 维数 & %d', cfg.numMfcc);
if cfg.useDelta
    fprintf(fid, ' + Delta');
end
if cfg.useDeltaDelta
    fprintf(fid, ' + DeltaDelta');
end
fprintf(fid, ' \\\\\n');
fprintf(fid, 'DTW 距离度量 & %s \\\\\n', cfg.dtwDistanceMetric);
fprintf(fid, '总体准确率 & %.2f\\%% (%d/%d) \\\\\n', stats.overallAccuracy, stats.numCorrect, stats.numTests);
fprintf(fid, '运行时间 & %.2f s \\\\\n', stats.runtimeSec);
fprintf(fid, '\\bottomrule\n\\end{tabular}\n\\end{table}\n');
end

function write_latex_class_accuracy_table(filePath, classAccTbl)
% 写出各数字类别识别准确率表。
fid = fopen(filePath, 'w', 'n', 'UTF-8');
if fid < 0
    return;
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '\\begin{table}[H]\n\\centering\n');
fprintf(fid, '\\caption{各数字类别识别准确率}\n');
fprintf(fid, '\\begin{tabular}{rrrr}\n\\toprule\n数字 & 测试数 & 正确数 & 准确率 \\\\\n\\midrule\n');
for k = 1:height(classAccTbl)
    fprintf(fid, '%d & %d & %d & %.2f\\%% \\\\\n', classAccTbl.Digit(k), ...
        classAccTbl.Total(k), classAccTbl.Correct(k), classAccTbl.AccuracyPercent(k));
end
fprintf(fid, '\\bottomrule\n\\end{tabular}\n\\end{table}\n');
end

function write_latex_confusion_table(filePath, confMat, digits)
% 写出十分类识别混淆矩阵表。
fid = fopen(filePath, 'w', 'n', 'UTF-8');
if fid < 0
    return;
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '\\begin{table}[H]\n\\centering\n\\scriptsize\n');
fprintf(fid, '\\caption{十分类识别混淆矩阵，行表示真实数字，列表示预测数字}\n');
fprintf(fid, '\\begin{tabular}{c');
for k = 1:numel(digits)
    fprintf(fid, 'r');
end
fprintf(fid, '}\n\\toprule\n真实/预测');
for k = 1:numel(digits)
    fprintf(fid, ' & %d', digits(k));
end
fprintf(fid, ' \\\\\n\\midrule\n');
for r = 1:numel(digits)
    fprintf(fid, '%d', digits(r));
    for c = 1:numel(digits)
        fprintf(fid, ' & %d', confMat(r, c));
    end
    fprintf(fid, ' \\\\\n');
end
fprintf(fid, '\\bottomrule\n\\end{tabular}\n\\end{table}\n');
end

function copy_report_assets(cfg)
% 复制图像、表格和核心代码文件到辅助材料目录。
figureNames = {'dataset_digit_counts.png', 'preprocess_example.png', 'mfcc_heatmap_example.png', ...
    'class_distance_bar_example.png', 'dtw_example_local_distance.png', ...
    'dtw_example_accumulated_path.png', 'dtw_example_mfcc_sequence_compare.png', ...
    'confusion_matrix.png', 'class_accuracy_bar.png'};
for k = 1:numel(figureNames)
    src = fullfile(cfg.figuresDir, figureNames{k});
    dst = fullfile(cfg.latexFiguresDir, figureNames{k});
    if exist(src, 'file')
        copyfile(src, dst, 'f');
    end
end

tableFiles = {'prediction_results.csv', 'class_accuracy.csv', 'confusion_matrix.csv'};
for k = 1:numel(tableFiles)
    src = fullfile(cfg.tablesDir, tableFiles{k});
    dst = fullfile(cfg.latexTablesDir, tableFiles{k});
    if exist(src, 'file')
        copyfile(src, dst, 'f');
    end
end

copyfile(fullfile(cfg.projectDir, 'main_topicE_dtw_speech_recognition.m'), cfg.latexCodeDir, 'f');
copyfile(fullfile(cfg.projectDir, 'config_topicE.m'), cfg.latexCodeDir, 'f');
funcFiles = dir(fullfile(cfg.functionsDir, '*.m'));
for k = 1:numel(funcFiles)
    copyfile(fullfile(funcFiles(k).folder, funcFiles(k).name), cfg.latexCodeDir, 'f');
end
end

function out = ternary(cond, a, b)
% 根据条件返回两个候选字符串中的一个，用于命令行结果显示。
if cond
    out = a;
else
    out = b;
end
end
