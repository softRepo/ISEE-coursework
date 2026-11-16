# 喇叭天线测量数据处理

`原始数据/` 保存 S11（驻波比）、S21、水平与垂直方向图、常规极化、栅网极化和距离测量的 7 份 CSV 数据。S11、S21 的频率列单位为 MHz，处理时换算为 GHz。

```sh
python -m pip install -r requirements.txt
python process_data.py
```

结果写入 `结果/`，包含频率响应、方向图、极化曲线、距离拟合图和数据摘要。`processed_summary.txt` 为现有数据的计算结果。
