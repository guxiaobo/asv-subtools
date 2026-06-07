"""
增量训练核心逻辑。

处理从 SQLite 读取的数据加载、模型 checkpoint 加载、
说话人 head 扩展和训练循环。

实际训练过程需要 ASV-Subtools v1 或 v2 框架支持。
此处提供骨架和接口定义。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from train.db import get_active_model, get_connection
from train.config import load_config

logger = logging.getLogger("train.trainer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_EMBEDDING_DIM = 192


class IncrementalTrainer:
    """
    增量训练器。

    负责：
    - 加载现有 checkpoint
    - 扩展 classifier head（新增说话人）
    - 在增量数据集上训练 1-3 epoch
    - 返回训练统计信息
    """

    def __init__(
        self,
        checkpoint_path: str,
        backbone: str = "CAM++",
        embedding_dim: int = 192,
        config: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            checkpoint_path: 基线模型 checkpoint 路径。
            backbone: 骨干网络名称（如 CAM++, ECAPA-TDNN）。
            embedding_dim: Embedding 维度。
            config: 训练配置字典。
        """
        self.checkpoint_path = checkpoint_path
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.config = config or {}

        self.base_lr = self.config.get("base_lr", 0.0001)
        self.epochs = self.config.get("epochs", 3)
        self.batch_size = self.config.get("batch_size", 64)

    def load_checkpoint(self) -> bool:
        """
        加载现有 checkpoint。

        Returns:
            是否成功加载。
        """
        ckpt = Path(self.checkpoint_path)
        if not ckpt.exists():
            logger.warning("Checkpoint 不存在: %s", self.checkpoint_path)
            return False

        logger.info("加载 checkpoint: %s", self.checkpoint_path)

        # TODO: 接入 ASV-Subtools 的模型加载
        # model = torch.load(self.checkpoint_path)
        # self.model = model
        # self.num_speakers = model.output_layer.weight.shape[-1]

        self.num_speakers = 100  # Placeholder
        return True

    def expand_classifier(self, num_new_speakers: int) -> int:
        """
        扩展分类器 head，为新增说话人添加权重。

        Args:
            num_new_speakers: 新增的说话人数量。

        Returns:
            扩展后的总说话人数。
        """
        old_n = getattr(self, "num_speakers", 0)
        new_n = old_n + num_new_speakers

        logger.info(
            "扩展 classifier: %d → %d (新增 %d 个说话人)",
            old_n, new_n, num_new_speakers,
        )

        # TODO: 接入 ASV-Subtools 的 expand_classifier
        # old_weight = model.output_layer.weight  # [D, old_N]
        # new_weight = torch.zeros(D, new_n)
        # new_weight[:, :old_N] = old_weight
        # nn.init.xavier_normal_(new_weight[:, old_n:])
        # model.output_layer.weight = nn.Parameter(new_weight)

        self.num_speakers = new_n
        return new_n

    def train(
        self,
        train_data_dir: str,
        num_new_speakers: int,
    ) -> Dict:
        """
        执行一轮增量训练。

        Args:
            train_data_dir: 增量训练数据目录（Kaldi 格式）。
            num_new_speakers: 新增的说话人数。

        Returns:
            训练统计信息。
        """
        start_time = time.time()

        # Load checkpoint
        if not self.load_checkpoint():
            logger.warning("Checkpoint 加载失败，可能使用随机初始化")

        # Expand head
        total_speakers = self.expand_classifier(num_new_speakers)

        # TODO: 接入 ASV-Subtools 的训练流程
        # 1. 创建 KaldiDataLoader
        # 2. 设置优化器 (lr = base_lr * 0.1)
        # 3. 训练循环
        # trainer_online.py --checkpoint <ckpt> --data-dir <train_data> --num-epochs 3

        duration = time.time() - start_time

        logger.info(
            "增量训练完成: %d 说话人, %.1f 秒",
            total_speakers, duration,
        )

        return {
            "total_speakers": total_speakers,
            "new_speakers": num_new_speakers,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.base_lr,
            "duration_sec": round(duration, 1),
        }

    def save_checkpoint(self, output_path: str) -> str:
        """
        保存训练后的 checkpoint。

        Args:
            output_path: 保存路径。

        Returns:
            绝对路径。
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # TODO: torch.save(model.state_dict(), output_path)
        logger.info("Checkpoint 已保存: %s", output_path)

        return str(path)
