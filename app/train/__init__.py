"""
ASV-Subtools 训练模块。

包含录音预处理 (preprocess) 和增量训练 (incremental_train) 两个 CLI 工具，
通过 SQLite 数据库与 API 服务串联，形成完整的数据收集 → 预处理 → 训练 → 发布流水线。
"""

__version__ = "0.1.0"
