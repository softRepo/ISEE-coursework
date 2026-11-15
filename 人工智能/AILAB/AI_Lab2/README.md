# 实验二：对弈搜索

- `debug.py`：3×3 井字棋 Minimax 实现。
- `alpha_beta.py`：4×4 对弈 GUI，使用 Alpha–Beta 搜索。
- `compare.py`：比较同一残局下 Minimax 与受限 Alpha–Beta 的耗时和搜索节点数。

```powershell
python -m pip install -r requirements.txt
python debug.py
python alpha_beta.py
python compare.py
```

GUI 需要 Python 的 Tk 支持。无限制 Minimax 的耗时随残局空位数增加。

[实验报告](report/report.pdf)
