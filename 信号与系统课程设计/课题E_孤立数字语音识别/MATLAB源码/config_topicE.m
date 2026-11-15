function cfg = config_topicE()
%CONFIG_TOPICE 课题E语音识别系统的集中配置文件。
% 本文件集中设置数据路径、样本划分、MFCC 参数、DTW 参数和输出控制参数。
% 修改参数后重新运行主脚本，即可按新的配置生成识别结果。

projectDir = fileparts(mfilename('fullpath'));
topicDir = fileparts(projectDir);

cfg.projectDir = projectDir;
cfg.topicDir = topicDir;
cfg.datasetDir = fullfile(projectDir, 'dataset');
cfg.recordingsDir = fullfile(cfg.datasetDir, 'recordings');

cfg.functionsDir = fullfile(projectDir, 'functions');
cfg.resultsDir = fullfile(projectDir, 'results');
cfg.figuresDir = fullfile(cfg.resultsDir, 'figures');
cfg.tablesDir = fullfile(cfg.resultsDir, 'tables');
cfg.featuresDir = fullfile(cfg.resultsDir, 'features');
cfg.logsDir = fullfile(cfg.resultsDir, 'logs');
cfg.reportDir = fullfile(projectDir, 'report');
cfg.latexDir = fullfile(cfg.reportDir, 'latex_report');
cfg.latexFiguresDir = fullfile(cfg.latexDir, 'figures');
cfg.latexTablesDir = fullfile(cfg.latexDir, 'tables');
cfg.latexCodeDir = fullfile(cfg.latexDir, 'code');

% 数据集与划分参数：编号 0-4 为测试样本，编号 5-49 为训练候选样本。
cfg.digits = 0:9;
cfg.testIndexList = 0:4;
cfg.trainIndexList = 5:49;

% 运行效率与模板选择参数。
cfg.maxTemplatesPerDigit = 30;
cfg.maxTestsPerDigit = inf;
cfg.templateSelectionMode = 'balanced_by_speaker';
cfg.classDecisionMode = 'min';       % 'min' 或 'mean_k_min'
cfg.meanK = 3;
cfg.randomSeed = 42;

% 音频预处理参数，用于统一采样率、分帧和端点检测。
cfg.targetFs = 8000;
cfg.enablePreEmphasis = true;
cfg.preEmphasisAlpha = 0.97;
cfg.frameLengthMs = 25;
cfg.frameShiftMs = 10;
cfg.endpointGuardFrames = 3;
cfg.endpointEnergyRatio = 0.12;
cfg.endpointZcrRatio = 0.35;
cfg.endpointSmoothFrames = 3;
cfg.minSpeechDurationSec = 0.08;

% MFCC 特征参数。输出矩阵方向统一为：行表示时间帧，列表示特征维度。
cfg.nfft = 256;
cfg.numMelFilters = 26;
cfg.numMfcc = 13;
cfg.useC0 = true;
cfg.useDelta = true;
cfg.useDeltaDelta = false;
cfg.deltaWindow = 2;
cfg.useCepstralMeanVarianceNorm = true;
cfg.melLowHz = 0;
cfg.melHighHz = cfg.targetFs / 2;

% DTW 匹配参数。默认使用欧氏距离，并关闭 Sakoe-Chiba 带约束。
cfg.dtwDistanceMetric = 'euclidean';  % 'euclidean' 或 'cosine'
cfg.useSakoeChibaBand = false;
cfg.sakoeChibaRatio = 0.20;

% 结果保存与绘图参数。
cfg.saveFeatureMat = true;
cfg.figureVisible = 'off';
cfg.figureDpi = 180;
cfg.closeFiguresAfterSave = true;
cfg.saveFigFile = false;
cfg.exampleTestOrder = 1;

end
