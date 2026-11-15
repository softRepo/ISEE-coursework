# 实验二：对弈搜索

- `debug.py`：3×3 井字棋 Minimax 的修复示例。
- `alpha_beta.py`：4×4 对弈 GUI，包含 Alpha–Beta 搜索及规则处理。
- `compare.py`：同一残局中无限制 Minimax 与受限 Alpha–Beta 的耗时、节点数对比。

```powershell
python -m pip install -r requirements.txt
python debug.py
python alpha_beta.py
python compare.py
```

GUI 需要 Python 的 Tk 支持；命令行程序与比较脚本可单独运行。无限制搜索的耗时取决于残局空位数。原算法和评估逻辑保留。

[脱敏实验报告](report/report.pdf)已删除身份栏和原始对话、终端及 GUI 截图，保留正文与实验数据。
