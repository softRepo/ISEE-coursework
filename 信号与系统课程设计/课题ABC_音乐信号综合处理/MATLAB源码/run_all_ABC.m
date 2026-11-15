clear; clc; close all;

%RUN_ALL_ABC 课题A、课题B和课题C综合实验总入口
%   本脚本负责初始化工程路径和统一配置，依次调用三个课题模块。
%   每个模块独立捕获运行异常，运行状态记录到 results/logs 目录。

rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);
addpath(fullfile(rootDir, 'common'));
addpath(fullfile(rootDir, 'topic_A'));
addpath(fullfile(rootDir, 'topic_B'));
addpath(fullfile(rootDir, 'topic_C'));

% 读取统一实验参数，并固定随机种子以保证噪声实验可复现。
cfg = config_ABC_experiments();
rng(cfg.rngSeed);

% 创建运行日志，用于记录各课题模块的执行状态。
logsDir = fullfile(rootDir, 'results', 'logs');
if ~exist(logsDir, 'dir')
    mkdir(logsDir);
end
logPath = fullfile(logsDir, 'run_all_ABC_log.txt');
[fid, msg] = fopen(logPath, 'w', 'n', 'UTF-8');
if fid < 0
    error('无法创建日志文件：%s。系统提示：%s', logPath, msg);
end
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

log_line(fid, '开始运行课题A、B、C综合实验。');
log_line(fid, '工程目录：%s', rootDir);

tasks = {
    '课题A', @() main_A_music_effect_analysis(rootDir, cfg);
    '课题B', @() main_B_sampling_reconstruction(rootDir, cfg);
    '课题C', @() main_C_noise_filter_design(rootDir, cfg)
    };

% 依次运行三个课题模块，单个模块失败时继续执行后续模块。
success = false(size(tasks, 1), 1);
messages = strings(size(tasks, 1), 1);

for iTask = 1:size(tasks, 1)
    taskName = tasks{iTask, 1};
    taskFcn = tasks{iTask, 2};
    log_line(fid, '----------------------------------------');
    log_line(fid, '开始%s。', taskName);
    try
        taskFcn();
        success(iTask) = true;
        messages(iTask) = "成功";
        log_line(fid, '%s运行成功。', taskName);
    catch ME
        success(iTask) = false;
        messages(iTask) = string(ME.message);
        log_line(fid, '%s运行失败：%s', taskName, ME.message);
        fprintf(2, '%s运行失败：%s\n', taskName, ME.message);
    end
end

copy_core_code_to_report(rootDir);

log_line(fid, '----------------------------------------');
log_line(fid, '运行总结：A=%s，B=%s，C=%s。', status_text(success(1)), status_text(success(2)), status_text(success(3)));
log_line(fid, '结果保存位置：%s', fullfile(rootDir, 'results'));
log_line(fid, '报告素材位置：%s', fullfile(rootDir, 'report', 'latex_report'));

fprintf('\n课题A：%s\n', messages(1));
fprintf('课题B：%s\n', messages(2));
fprintf('课题C：%s\n', messages(3));
fprintf('结果目录：%s\n', fullfile(rootDir, 'results'));
fprintf('报告目录：%s\n', fullfile(rootDir, 'report'));
fprintf('运行日志：%s\n', logPath);

function log_line(fid, fmt, varargin)
%LOG_LINE 向命令行和日志文件写入带时间戳的信息
msg = sprintf(fmt, varargin{:});
lineText = sprintf('[%s] %s', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), msg);
fprintf('%s\n', lineText);
fprintf(fid, '%s\n', lineText);
end

function txt = status_text(ok)
%STATUS_TEXT 将逻辑状态转换为中文运行状态
if ok
    txt = '成功';
else
    txt = '失败';
end
end

function copy_core_code_to_report(rootDir)
%COPY_CORE_CODE_TO_REPORT 复制核心代码文件到配套材料目录
codeDir = fullfile(rootDir, 'report', 'latex_report', 'code');
if ~exist(codeDir, 'dir')
    mkdir(codeDir);
end
files = {
    fullfile(rootDir, 'run_all_ABC.m')
    fullfile(rootDir, 'config_ABC_experiments.m')
    fullfile(rootDir, 'topic_A', 'main_A_music_effect_analysis.m')
    fullfile(rootDir, 'topic_B', 'main_B_sampling_reconstruction.m')
    fullfile(rootDir, 'topic_C', 'main_C_noise_filter_design.m')
    };
for iFile = 1:numel(files)
    if exist(files{iFile}, 'file')
        copyfile(files{iFile}, fullfile(codeDir, get_file_name(files{iFile})));
    end
end
end

function name = get_file_name(filePath)
[~, base, ext] = fileparts(filePath);
name = [base, ext];
end
