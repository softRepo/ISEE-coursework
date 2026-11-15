function [predictedDigit, classDistances, detail] = classify_by_dtw(testFeatures, templates, cfg)
%CLASSIFY_BY_DTW 使用 DTW 模板匹配完成 0-9 孤立数字分类。
% 输入测试样本特征和模板特征，按类别汇总 DTW 距离并输出预测标签。

digits = cfg.digits(:).';
numClasses = numel(digits);
numTemplates = numel(templates);
allDistances = inf(numTemplates, 1);

bestDistance = inf;
bestTemplateIndex = NaN;

for k = 1:numTemplates
    [dist, ~] = dtw_distance_manual(testFeatures, templates(k).features, cfg, false);
    allDistances(k) = dist;
    if dist < bestDistance
        bestDistance = dist;
        bestTemplateIndex = k;
    end
end

classDistances = inf(1, numClasses);
for c = 1:numClasses
    digit = digits(c);
    templateDigits = [templates.digit];
    idx = find(templateDigits == digit);
    distances = allDistances(idx);
    distances = distances(isfinite(distances));

    if isempty(distances)
        classDistances(c) = inf;
    elseif strcmpi(cfg.classDecisionMode, 'mean_k_min')
        distances = sort(distances, 'ascend');
        kMean = min(cfg.meanK, numel(distances));
        classDistances(c) = mean(distances(1:kMean));
    else
        classDistances(c) = min(distances);
    end
end

[~, bestClassIdx] = min(classDistances);
predictedDigit = digits(bestClassIdx);

detail = struct();
detail.allTemplateDistances = allDistances;
detail.bestTemplateIndex = bestTemplateIndex;
detail.bestTemplateDistance = bestDistance;
detail.classDistances = classDistances;

end
