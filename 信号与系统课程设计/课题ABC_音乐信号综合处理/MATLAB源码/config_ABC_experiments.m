function cfg = config_ABC_experiments()
%CONFIG_ABC_EXPERIMENTS 课题A、课题B和课题C实验参数配置
%   本文件集中设置音效分析、采样重构和加噪滤波实验的关键参数。
%   修改参数后重新运行 run_all_ABC.m 或对应课题脚本即可更新结果。

cfg = struct();
cfg.targetFs = 44100;
cfg.enablePlayback = false;
cfg.saveFigures = true;
cfg.showFigures = false;
cfg.rngSeed = 2026;
cfg.figureResolution = 220;
cfg.maxSpectrumFreqHz = 10000;
cfg.localWaveSeconds = 5;

% 输入音频片段文件名。三个文件应位于 segments 目录。
cfg.segments.A = 'A_fur_elise_segment.wav';
cfg.segments.B = 'B_hotel_california_segment.wav';
cfg.segments.C = 'C_faded_segment.wav';

% 课题A参数：回声、混响和STFT窗长设置。
cfg.A.echoDelaySec = 0.35;
cfg.A.echoAlpha = 0.45;
cfg.A.reverbDelaysSec = [0.05, 0.09, 0.14, 0.21];
cfg.A.reverbGains = [0.45, 0.30, 0.22, 0.15];
cfg.A.stftShortWinMs = 25;
cfg.A.stftLongWinMs = 80;
cfg.A.stftHopRatio = 0.5;

% 课题B参数：有效带宽估计、采样率设计和局部波形显示范围。
cfg.B.energyRatioForBandwidth = 0.99;
cfg.B.dbDropForBandwidth = 40;
cfg.B.oversamplingFactor = 2.5;
cfg.B.criticalFactor = 1.05;
cfg.B.undersamplingFactor = 0.65;
cfg.B.displayTimeRangeSec = [0, 8];
cfg.B.minSamplingRate = 4000;
cfg.B.maxSamplingRate = cfg.targetFs;

% 课题C参数：噪声强度、单频干扰频率和滤波器参数。
cfg.C.whiteNoiseSNRdB = 12;
cfg.C.singleToneNoiseAmp = 0.08;
cfg.C.noiseFreqHz = 3000;
cfg.C.lowpassOrders = [4, 6, 8];
cfg.C.lowpassCutoffsHz = [2500, 4000, 6000];
cfg.C.notchBandwidthHz = 80;
cfg.C.defaultLowpassOrder = 6;
cfg.C.defaultLowpassCutoffHz = 4000;
end
