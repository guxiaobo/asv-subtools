# 每日坐席录音声纹模型优化方法论

> **目标：** 使用每日坐席录音，通过 VAD 拆分、说话人标注、增量训练三步，
> 持续优化三个声纹模型（CAM++ / ECAPA-TDNN / ResNet34）的客户区分能力。

---

## 1. 工作流总览

```
每日录音 .wav
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ Phase 1: 预处理                                             │
│  1. 加载录音、降噪、SNR 评估                                 │
│  2. VAD 能量切割（lossless） → 语音段 .wav                   │
│  3. 选配：说话人标注（坐席 vs 客户）+ 声纹入库               │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ Phase 2: 跨录音聚合（说话人打标）                              │
│  1. 同客户多通录音合并 — Otsu 分割 + pairwise 聚类            │
│  2. 客户 centroid 声纹入库                                   │
│  3. Agent 声纹更新（跨录音汇总）                              │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ Phase 3: 增量训练                                           │
│  1. 从 DB 读取标注好的客户段                                  │
│  2. 加载预训练 backbone + 新分类头                            │
│  3. Fine-tune 5~20 epoch                                    │
│  4. backbone 权重入库 + 模型版本注册                          │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ Phase 4: 评估对比                                           │
│  1. 提取所有段 pretrained vs fine-tuned 的 embedding          │
│  2. 计算 within/between-class cosine similarity              │
│  3. Per-speaker 逐用户分析                                     │
└────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1: 预处理（VAD 语音拆分）

**入口脚本:** `app/train/preprocess.py`  
**VAD 核心函数:** `app/train/vad.py:energy_vad()`（第 48 行）

### 2.1 命令

```bash
# 处理所有待处理录音
cd app && PYTHONPATH=. /opt/anaconda3/bin/python -m train.preprocess

# 限处理 100 条
python -m train.preprocess --limit 100

# 仅催收业务线
python -m train.preprocess --biz collection

# 处理单条
python -m train.preprocess --call-id COL20250601_001

# 持续监听模式（每 60s 轮询新录音）
python -m train.preprocess --watch

# 仅检查待处理清单
python -m train.preprocess --dry-run
```

### 2.2 VAD 算法详解

**`energy_vad()` — 基于能量的自适应阈值 VAD**

1. **过滤开头提示音（可选）** — `filter_leading_sec=2.0`
2. **帧能量计算** — 帧长 30ms，50% 重叠
3. **自适应阈值** — `noise_floor = P10 能量百分位`；`阈值 = noise_floor × 10^(threshold×10/10)`（至少 3dB）
4. **语音/静音标记** — 能量 > 阈值为语音帧
5. **合并断开的段** — gap < 0.8s 的段合并
6. **lossless 模式**（默认）— 保留所有段，不丢弃短段或低 SNR 段

**VAD 参数配置：**

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `threshold` | 0.5 | 噪声底噪之上的 dB 倍率 |
| `min_segment_sec` | 0.0 | 最小段长（lossless=False 有效） |
| `max_segment_sec` | 30.0 | 最大段长 |
| `filter_leading_sec` | 2.0 | 过滤开头提示音 |
| `lossless` | true | 是否保留所有分段 |

### 2.3 预处理完整流程

`preprocess_recording()`（第 282 行）一次调用完成：

```
audio_path → load_audio (16kHz mono)
    → reduce_noise (降噪，仅在音频 >1s)
    → estimate_snr (整体 SNR)
    → energy_vad (lossless 切割)
    → save_segments (保存到 preprocessed/collection/<date>/<call_id>/)
    → 可选: SpeakerDiarizer.diarize (说话人标注)
```

**输出产物：**
- `app/data/preprocessed/collection/YYYY-MM-DD/<call_id>/<call_id>_seg_NNNN.wav`
- `audio_segments` 表记录（path, start, end, duration, diarization label）
- `speaker_voiceprints` 表记录（若跑标注）

---

## 3. Phase 2: 说话人打标 - SpeakerDiarizer

**核心文件:** `app/train/diarizer.py`  
**入口:** `--cross-aggregate-only` 标志（仅执行 Phase 2，跳过 Phase 1）

### 3.1 命令触发

```bash
# 在预处理后执行跨录音聚合
python -m train.preprocess --cross-aggregate-only

# 对现有 VAD 段重新跑说话人标注（独立使用）
cd app && PYTHONPATH=. /opt/anaconda3/bin/python -m train.diarizer --input ./data/preprocessed

# 从 WEB UI 触发
浏览器 → 模型管理主页 → 点击「🏷️ 说话人打标」按钮
```

### 3.2 标注算法（v4: Otsu 双峰分割）

`SpeakerDiarizer.diarize()`（第 421 行）核心流程：

**Step 1: 提取 embedding**
- 逐个加载 VAD 段 WAV → 提取 80-dim log-mel FBank（纯 numpy 实现，无 librosa）
- ONNX Runtime 推理 → 192/256 维 embedding → L2 归一化

**Step 2: 计算与坐席声纹的相似度**
- 从 `speaker_voiceprints` 表加载对应模型的坐席声纹
- 余弦相似度：`dot(seg_emb, agent_ref)`

**Step 3: Otsu 双峰分割**
```
scores = [sim_to_agent for segment in segments]
best_t = argmin(W1·Var(G1) + W2·Var(G2))   # Otsu 准则
# G1 = scores ≤ t (客户候选), G2 = scores > t (坐席)
```
- 要求 ≥4 个有效段才启用 Otsu
- 阈值范围裁剪到 [0.30, 0.90]
- fallback：段数不足时用模型特定默认阈值

**Step 4: 客户段内 pairwise 聚类确认**
- 客户候选段之间计算成对余弦相似度矩阵
- 邻接矩阵：`sim >= cluster_threshold`（去掉对角线自环）
- 连通分量（BFS）聚类 → 取最大群组为主客户
- centroid embedding：群组内所有 embedding 平均后重归一化
- 标注主群组段为 `customer`，其余段为 `uncertain`

**Step 5: 多说话人检测**
- 第二大群组 ≥ 主群组 60% → 存在多说话人
- 多说话人场景下非主群组标记为 `uncertain`

### 3.3 模型特定阈值

| 模型 | 坐席判定阈值 | 客户聚类阈值 | 说明 |
|------|------------|------------|------|
| CAM++ | 0.49 | 0.35 | 电话录音客户段间相似度仅 0.25-0.50，需低阈值 |
| ResNet34 | 0.59 | 0.55 | 稳定中等阈值 |
| ECAPA | 0.68 | 0.55 | 高坐席阈值，倾向高置信度 |

### 3.4 跨录音声纹聚合

**`cross_call_aggregate()`**（第 864 行）

```
for each customer (by phone number):
    1. 收集所有通录音中非坐席段 (customer + uncertain + customer_candidate)
    2. 跨录音 pairwise 聚类 (阈值 = cluster_threshold)
    3. 取最大群组的 centroid 作为该客户声纹
    4. 写入 speaker_voiceprints 表 (speaker_type='customer')
```

- 最小聚类样本数：`min_samples=2`
- 只处理有 ≥2 个非坐席段的客户
- 入库字段：`embedding`, `segment_count`, `source_call_ids`

---

## 4. Phase 3: 增量训练（Fine-tune）

**核心文件:** `app/train/fine_tune.py`  
**入口函数:** `train_model()`（第 893 行）

### 4.1 命令

```bash
# 训练单个模型
cd app && PYTHONPATH=. /opt/anaconda3/bin/python -m train.fine_tune --model campplus

# 训练所有三个模型（顺序执行）
python -m train.fine_tune --model all

# 自定义参数
python -m train.fine_tune --model ecapa --epochs 20 --lr 1e-4 --batch-size 16 --max-frames 400

# 从 WEB UI 触发
浏览器 → 模型管理主页 → 点击「📈 增量训练」按钮
```

### 4.2 CLI 参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--model` | `all` | campplus / ecapa / resnet / all |
| `--epochs` | 5 | 训练轮数（推荐 20 取得最佳效果） |
| `--lr` | 1e-4 | backbone 学习率（分类头自动 ×10） |
| `--batch-size` | 16 | batch size（受 GPU 显存限制） |
| `--max-frames` | 400 | 每段最大帧数（4s @ 10ms） |

### 4.3 数据构建（第 745 行）

1. 查询 DB 中 `status='preprocessed'` 的录音
2. 从 `audio_segments` 表获取段路径
3. 对每通录音用 `SpeakerDiarizer` 做 Otsu 分割标注
4. 只保留 `customer` 标记段
5. 按 `customer_phone`（DB 字段）关联到说话人 ID
6. 主流程已包含标注，无需离线预跑

**旧数据兼容：** `_build_training_data_fs()`（第 829 行） — 当 `audio_segments` 表为空时
（预迁移场景），扫描 `preprocessed/collection/` 目录结构。

### 4.4 模型加载

```python
# CAM++
model = CAMPlus(feat_dim=80, embedding_dim=192, num_speakers=N)
state_dict = torch.load("pytorch_weights/campplus_cn_common.pt")
model.load_pretrained(state_dict)  # 跳过 projection 层

# ECAPA
model = ECAPA_TDNNSpeaker(feat_dim=80, embedding_dim=192, num_speakers=N)
state_dict = torch.load("pytorch_weights/avg_model.pt")
model.load_pretrained(state_dict)

# ResNet34
model = ResNet34_2D(feat_dim=80, embedding_dim=256, num_speakers=N)
state_dict = torch.load("pytorch_weights/avg_model")
model.load_pretrained(state_dict)
```

### 4.5 训练配置

| 配置项 | 值 |
|--------|-----|
| 设备 | CUDA（如有）/ 回退 CPU |
| 优化器 | Adam（两阶段：backbone 1e-4, classifier 1e-3） |
| 权重衰减 | 1e-4 |
| 学习率调度 | CosineAnnealingLR（T_max=epochs） |
| 损失函数 | CrossEntropyLoss |
| 梯度裁剪 | 5.0 |
| 训练/验证分割 | 每个说话人内 85%/15%（per-speaker 分层） |
| 数据增强 | 随机时间偏移（crop/pad 到 400 帧） |

### 4.6 训练输出

```
app/pytorch_weights/fine_tuned/
├── campplus_best.pt          # 最佳验证 epoch（完整权重）
├── campplus_final.pt         # 最后一 epoch（完整权重）
├── campplus_backbone.pt      # 去掉 projection（推理用）
├── ecapa_best.pt
├── ecapa_final.pt
├── ecapa_backbone.pt
├── resnet_best.pt
├── resnet_final.pt
└── resnet_backbone.pt
```

权重保存后自动注册 `model_versions` 表（版本标签、MD5、验证指标等）。

### 4.7 经验数据（20 epoch, 125 段/15 说话人）

| 模型 | 更新前 Separation | 更新后 Separation | 同说话人改善 |
|------|-----------------|-----------------|-------------|
| ResNet34 | 0.157 | 0.699 | 8/11 说话人 ↑ |
| ECAPA | 0.0015 (差) | 0.787 (强) | 大幅提升 |
| CAM++ | 0.050 | 0.290 (中等↑) | 同说话人 ↑ 但跨说话人↓ |

> **结论：** 少量数据（125段/15人）增量训练后，ECAPA 提升最大，
> ResNet34 最稳健，CAM++ 改善有限。持续积累标注数据后效果更佳。

---

## 5. Phase 4: 评估对比

**核心文件:** `app/train/evaluate.py`

### 5.1 命令

```bash
# 对比全部模型
cd app && PYTHONPATH=. /opt/anaconda3/bin/python -m train.evaluate

# 指定训练 epoch 数（用于匹配 best.pt 路径）
python -m train.evaluate --epoch 20

# 仅评估某个模型
python -m train.evaluate --model campplus

# 从 WEB UI 触发
浏览器 → 模型管理主页 → 点击「📊 模型评估对比」按钮
```

### 5.2 评估指标

**核心指标：Separation = Within-class mean - Between-class mean**

- **Within-class 余弦相似度：** 同说话人不同段之间的 cosine
- **Between-class 余弦相似度：** 不同说话人之间的 cosine
- **Separation（Δ）：** 两者的差值，越大越好
  - Separation > 0.5 → 良好的说话人区分度
  - Separation < 0.1 → 模型几乎不区分说话人

**Per-speaker 分析：** 逐说话人统计 within-class 相似度变化，标记 ↑ 改善 / ↓ 退化。

### 5.3 评估流程

```
1. 加载所有标注训练段（与训练同一数据集）
2. 对每个模型：
   a. 加载预训练 backbone，提取 embedding
   b. 计算 pairwise 相似度统计
   c. 加载 fine-tuned backbone，重复 a/b
   d. 对比 Δ：Separation fine-tuned - Separation pretrained
3. 输出汇总表：
   Model     Pretrained Sep    Fine-tuned Sep     Δ
   CAM++     0.050            0.290             +0.240
   ECAPA     0.0015           0.787             +0.786
   ResNet34  0.157            0.699             +0.542
```

### 5.4 CLI 参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--model` | `all` | campplus / ecapa / resnet / all |
| `--epoch` | 20 | 匹配 fine_tuned/<name>_backbone.pt（固定命名） |

---

## 6. 生产环境关键路径

### 6.1 一键全流程

```bash
# Step 1: 预处理新录音（VAD 切割）
cd /Users/guxiaobo/Documents/GitHub/asv-subtools/app
PYTHONPATH="$PWD" /opt/anaconda3/bin/python -m train.preprocess --limit 100

# Step 2: 跨录音聚合（说话人打标）— 标注+声纹入库
PYTHONPATH="$PWD" /opt/anaconda3/bin/python -m train.preprocess --cross-aggregate-only

# Step 3: 增量训练三个模型
PYTHONPATH="$PWD" /opt/anaconda3/bin/python -m train.fine_tune --model all --epochs 20

# Step 4: 评估对比
PYTHONPATH="$PWD" /opt/anaconda3/bin/python -m train.evaluate --epoch 20
```

### 6.2 WEB UI 对应操作

| WEB 按钮 | 对应脚本 | 操作说明 |
|---------|---------|---------|
| 🔄 批量预处理（VAD） | `train.preprocess` | 触发 Phase 1 |
| 🏷️ 说话人打标 | `train.preprocess --cross-aggregate-only` | 触发 Phase 2 |
| 📈 增量训练 | `train.fine_tune --model all` | 触发 Phase 3 |
| 📊 模型评估对比 | `train.evaluate --model all` | 触发 Phase 4 |
| 🚀 发布线上模型 | 标记 model_versions 某版本为 active | 线上切换 |

### 6.3 WEB UI 性能注意事项（2026-06-15 修复）

- ~~Bug: 按钮触发同步 `subprocess.run()` 阻塞 FastAPI 事件循环~~ ✅ 已修复
- 修复方式：所有 4 个 handler 改用 `asyncio.to_thread()` 异步执行
- 大训练集（>1000 段）下的超时限制见各 handler（120s~14400s）

### 6.4 数据库表结构

**关键表：**

| 表 | 用途 | 关键字段 |
|----|------|---------|
| `recordings` | 录音元数据 | `call_id`, `customer_phone`, `pre_status`, `pre_result` |
| `audio_segments` | VAD 段信息 | `recording_id`, `file_path`, `start_sec`, `end_sec`, `is_ignored` |
| `speaker_voiceprints` | 声纹库 | `model_name`, `speaker_type` (agent/customer), `embedding` |
| `model_versions` | 模型版本 | `model_name`, `version_tag`, `eval_value`, `model_path` |

### 6.5 常见问题

**Q: VAD 切割结果不稳定？**
- 检查 `conf/vad_config.json` 中的 threshold 参数
- 电话录音推荐 `threshold=0.45`（比默认 0.5 更灵敏）
- 背景噪声大的录音可降低 threshold

**Q: 说话人标注不准（全部标为坐席/全部标为客户）？**
- 检查 `speaker_voiceprints` 表中是否有坐席声纹记录
- 核查 call_id 格式是否规范（`<客户名>-<电话号码>` 模式）
- 尝试不同的 ONNX 模型（CAM++ / ResNet34 / ECAPA 各有适用的电话场景）

**Q: 增量训练后某个说话人效果反而变差？**
- 该说话人训练段太少（<3 段），过拟合了
- 增加该说话人的标注数据后重训
- 可以考虑降低 epochs 或提高 weight_decay

**Q: ONNX Runtime 报错/无法推理？**
- 确保 `app/api/models/` 目录下有对应的 `.onnx` 文件
- 模型文件清单：
  - `campplus.onnx`（默认，192-dim）
  - `voxceleb_resnet34_LM.onnx`（256-dim）
  - `ecapa-speaker-v1.onnx`（192-dim）

### 6.6 开发环境

- Python: 3.11.15（conda base）
- PyTorch: 2.10.0（CPU，Apple Silicon M4）
- ONNX Runtime: CPUExecutionProvider
- SQLite: `app/data/training.db`
- Web 服务: FastAPI + uvicorn（端口 8000）
