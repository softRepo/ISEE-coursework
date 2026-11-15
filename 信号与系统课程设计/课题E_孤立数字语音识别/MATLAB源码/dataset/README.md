# 语音数据

本课题使用 [Free Spoken Digit Dataset（FSDD）](https://github.com/Jakobovski/free-spoken-digit-dataset)。从原仓库下载 `recordings` 目录，放到本目录中：

```text
dataset/
├─ README.md
└─ recordings/
   └─ {digitLabel}_{speakerName}_{index}.wav
```

报告使用 6 位说话人的 3,000 条英文数字录音，采样率为 8 kHz。录音编号 `0–4` 用于测试，`5–49` 用作训练模板候选；每类最多选取 30 个模板。

数据集由 FSDD 项目提供，使用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 许可。
