# 模型优化管道：VAD → 说话人标记 → 增量训练 → 效果对比

> 更新时间：2026-06-19
> 涉及模块：`app/train/vad.py`、`app/train/diarizer.py`、`app/train/preprocess.py`、
> `app/train/fine_tune.py`、`app/train/trainer.py`、`app/train/evaluate.py`

---

## 目录

1. [优化管道总览](#1-优化管道总览)
2. [每日坐席录音 VAD 拆分](#2-每日坐席录音-vad-拆分)
3. [说话人标注（Diarizer）](#3-说话人标注diarizer)
4. [训练数据准备](#4-训练数据准备)
5. [增量训练（Fine-Tune）](#5-增量训练fine-tune)
6. [模型效果对比](#6-模型效果对比)
7. [模型版本管理与部署](#7-模型版本管理与部署)
8. [命令行工具](#8-命令行工具)

---

## 1. 优化管道总览

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 坐席录音文件  │ → │ VAD 语音拆分  │ → │ 说话人标注    │ → │ 训练数据入库  │
│ (WAV 16kHz)  │    │ (energy_vad) │    │ (Diarizer)   │    │ (SQLite + fs) │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                     ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 部署 ONNX    │ ← │ 评估对比      │ ← │ 增量训练      │
│ (热加载)      │    │ (evaluate.py)│    │ (fine_tune)  │
└──────────────┘    └──────────────┘    └──────────────┘
```

## 2. 每日坐席录音 VAD 拆分

**代码位置：** `app/train/vad.py`

### 2.1 核心函数：`energy_vad()`

```python
def energy_vad(waveform, sample_rate, lossless=False, **kwargs) -> List[Tuple[np.ndarray, int, int]]:
```

基于短时能量的活动语音检测（无依赖的纯 numpy 实现）。

**参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_ms` | 30 | 分帧窗长(ms) |
| `threshold` | 0.5 | VAD 能量阈值（0-1 归一化） |
| `lossless` | False | 无损模式：不丢弃任何段，过大段自动切分 |
| `min_segment_sec` | 1.5 | 最小时长(秒)，短于此丢弃 |
| `max_segment_sec` | 15.0 | 最大时长(秒)，长于此切分 |
| `filter_leading_sec` | 2.0 | 开头静音过滤时长 |
| `filter_prefix` | "" | 前缀过滤关键字 |

**处理流程：**

```
原始波形 (float32, 16kHz)
  │
  ├─ 前置静音过滤（filter_leading_sec）
  │
  ├─ 分帧: 30ms 帧长, 10ms 帧移
  │
  ├─ 计算帧级 RMS 能量
  │
  ├─ 归一化到 0-1（基于最大 RMS）
  │
  ├─ 能量阈值判决 → 活动/静音帧标记
  │
  ├─ 后处理:
  │   ├─ 合并静音间隙 < 300ms
  │   ├─ 丢弃过短段 (< min_segment_sec)
  │   ├─ 切分过长段 (> max_segment_sec)
  │   └─ lossless=True: 强制保留所有帧（语音/非语音都保留）
  │
  └─ 输出: [(waveform_segment, start_sample, end_sample), ...]
```

### 2.2 降噪

**代码位置：** `app/train/audio_utils.py` — `reduce_noise()`

使用频谱减法（spectral subtraction）对长录音降噪，基于非语音帧的噪声谱估计。每帧估计噪声谱，语音帧减去噪声谱。

### 2.3 预处理编排：`preprocess_recording()`

**代码位置：** `app/train/vad.py:282–457`

单条录音的完整预处理入口点，由 `preprocess.py` 或 Web 触发器调用：

```python
def preprocess_recording(
    audio_path, output_dir,
    sample_rate=16000,
    channel_separated=False,
    apply_noise_reduction=True,
    run_diarization=True,
    diarizer_model_name="CAM++",
    diarizer_db_path=None,
    diarizer_agent_threshold=None,
    diarizer_cluster_threshold=0.55,
    **vad_kwargs,
) -> dict:
```

**处理链：**

```
load_audio(audio_path) → reduce_noise() → energy_vad(lossless=True)
  → save_segments(output_dir, prefix, wav_list)
  → (可选) SpeakerDiarizer.diarize(segment_paths)
  → (可选) get_customer_voiceprint()
  → 返回统计 dict（段数、SNR、说话人分布）
```

**输出目录结构：**

```
{preprocessed_root}/{biz_system}/{date_str}/{call_id}/
  ├── seg_001.wav
  ├── seg_002.wav
  ├── ...
  └── seg_NNN.wav
```

**返回结果结构：**

```python
{
    "segment_count": 12,           # 总段数
    "agent_segments": 6,           # 标注为坐席的段数
    "customer_segments": 4,        # 标注为客户段数
    "uncertain_segments": 2,       # 未确定段数
    "agent_valid_sec": 45.2,       # 坐席有效语音时长(秒)
    "customer_valid_sec": 31.8,    # 客户有效时长
    "avg_snr_db": 18.5,
    "diarization_done": True,
    "customer_voiceprint_available": True,
    "customer_voiceprint_num_segments": 4,
    "segment_details": [            # 每段元数据
        {"file_path": "...", "start_sec": 0.0, "end_sec": 2.3, "duration_sec": 2.3},
        ...
    ],
}
```

### 2.4 命令行入口

```bash
# 处理待处理录音
python -m train.preprocess --limit 50

# 处理指定业务
python -m train.preprocess --limit 10 --biz collection

# 处理单条
python -m train.preprocess --call-id CALL001

# 持续监听模式（每 60s 轮询）
python -m train.preprocess --watch

# 跳过说话人标注（仅 VAD）
python -m train.preprocess --no-diarize

# 指定说话人标注模型
python -m train.preprocess --diarizer-model ResNet34
```

---

## 3. 说话人标注（Diarizer）

**代码位置：** `app/train/diarizer.py`

### 3.1 核心类：`SpeakerDiarizer`

```python
class SpeakerDiarizer:
    def __init__(self, model_path=None, db_path=None, model_name="CAM++",
                 agent_threshold=None, cluster_threshold=0.55):
```

### 3.2 说话人标注流程

```
VAD 切割后的段列表 [seg_1.wav, seg_2.wav, ..., seg_N.wav]
  │
  ├─ Phase 1: 双通道录音的说话人分离
  │   ├─ 通道分离检测 (detect_double_talk)
  │   ├─ 双讲段移除
  │   └─ 坐席/客户通道信号分离
  │
  ├─ Phase 2: 说话人 embedding 提取
  │   └─ 对每段调用 ONNX 或 PyTorch 模型提取 d-vector
  │
  ├─ Phase 3: Otsu 坐席筛选
  │   ├─ 使用参考声纹（agent_ref embedding）做余弦相似度
  │   ├─ agent_threshold=None → 动态阈值检测（Otsu 双峰法）
  │   ├─ 高于阈值的段 → 坐席（agent）
  │   └─ 剩余段进入客户聚类
  │
  ├─ Phase 4: 客户聚类
  │   ├─ Cluster 算法：亲和度传播（AffinityPropagation）
  │   │   或 HDBSCAN（取决于参数）
  │   ├─ cluster_threshold=0.35, min_samples=2
  │   └─ 各 cluster 计算 centroid embedding
  │
  └─ Phase 5: 标签输出
      └─ 每段分配 speaker_label + speaker_type（agent/customer）
```

### 3.3 客户声纹中心（centroid）

**代码位置：** `diarizer.py:815` — `get_customer_voiceprint()`

```python
def get_customer_voiceprint(self, results, seg_paths):
```

对同一说话人 ID 的所有段，计算其 embedding 的均值作为 centroid 声纹：

```python
def _compute_centroid(embeddings):
    """计算 centroid embedding = stack → mean → L2-normalize"""
    stacked = F.normalize(torch.stack(embeddings), dim=1)
    centroid = stacked.mean(dim=0)
    centroid = F.normalize(centroid, dim=0)
    return centroid.numpy()
```

**存储位置：** `speaker_voiceprints` 表

| 列 | 说明 |
|----|------|
| model_name | 声纹模型名称 |
| speaker_type | 'agent' 或 'customer' |
| speaker_id | 唯一说话人 ID |
| embedding | numpy float32 序列化 BLOB |
| segment_count | 构成该 centroid 的段数 |
| source_call_ids | 来源通话 ID 列表 (JSON) |

### 3.4 声纹相似度计算

```python
# app/api/services/verifier.py
emb_A = extract_embedding(wav_A)   # (dim,)  L2 归一化
emb_B = extract_embedding(wav_B)   # (dim,)  L2 归一化
score = float(emb_A @ emb_B)       # 余弦相似度（在 L2 归一化后等于点积）
decision = 'same' if score >= threshold else 'different'
```

- 所有 embedding 都经过 L2 归一化，余弦相似度退化为点积
- 三个模型统一输出 L2 归一化的 d-vector
- 阈值由业务场景配置（默认 0.5 ~ 0.7 视安全级别而定）

---

## 4. 训练数据准备

**代码位置：** `app/train/fine_tune.py:751–854` — 函数 `build_training_data()`

### 4.1 数据源

三种来源优先级：

1. **SQLite DB**（首选）— 从 `audio_segments` 表查询标注好的段，标签来源 `speaker_label`
2. **文件系统**（备选）— 扫描已标注的目录结构
3. **手动指定**（调试）— 直接传入目录路径

### 4.2 数据加载类：`SpeakerDataset`

```python
class SpeakerDataset(Dataset):
    def __init__(self, db_path_or_dir, model_name="CAM++", max_frames=400):
        # 构建 segments = [(wav_path, speaker_id), ...]
        # 每个说话人分组，确保至少 2 段/人（交叉熵损失需要支持多类）

    def __getitem__(self, idx):
        wav_path, speaker_id = self.segments[idx]
        feat = self.fbank.extract_from_file(wav_path)   # (T, 80)
        # 中心裁剪 / 补零到 max_frames
        if feat.size(0) > max_frames:
            start = (feat.size(0) - max_frames) // 2
            feat = feat[start:start + max_frames]
        else:
            pad = max_frames - feat.size(0)
            feat = F.pad(feat, (0, 0, 0, pad))
        return feat, speaker_id
```

### 4.3 FBank 特征提取

**代码位置：** `fine_tune.py:642–710` — 类 `FBankExtractor`

纯 torch 实现，零外部依赖：

```python
class FBankExtractor:
    def __init__(self, n_mels=80, n_fft=512, hop_length=160, win_length=400, sr=16000):
        # STFT → power spectrum → Mel filterbank (80 bins, 0–8kHz) → log

    def extract_from_file(self, wav_path):
        wav, sr = soundfile.read(wav_path)  # 或 torchaudio
        return self.extract(torch.from_numpy(wav).float())

    def extract(self, wav):
        spec = torch.stft(wav, n_fft, hop_length, win_length,
                          window=torch.hann_window(win_length),
                          return_complex=True)
        power = spec.abs() ** 2
        mel = power @ self.mel_weights  # (F, 80) 矩阵乘法
        feat = torch.log(torch.clamp(mel, min=1e-12))
        return feat.T  # (T, 80)
```

### 4.4 数据划分

```
1. 按说话人分层采样
2. 80% 训练 / 20% 验证（按说话人分层）
3. 每个说话人至少 2 段训练段
4. 不足 2 段的说话人从验证集补足
```

---

## 5. 增量训练（Fine-Tune）

**代码位置：** `app/train/fine_tune.py:856–1171` — 函数 `fine_tune_model()`

### 5.1 初始化

```python
# 根据 model_name 选择架构
if model_name == "CAM++":
    model = CAMPlus(feat_dim=80, embedding_dim=192, num_speakers=num_speakers)
    pretrained = WEIGHTS_DIR / "campplus_cn_common.pt"
elif model_name == "ECAPA":
    model = ECAPA_TDNNSpeaker(feat_dim=80, embedding_dim=192, num_speakers=num_speakers)
    pretrained = WEIGHTS_DIR / "avg_model.pt"
elif model_name == "ResNet34":
    model = ResNet34_2D(feat_dim=80, embedding_dim=256, num_speakers=num_speakers)
    pretrained = WEIGHTS_DIR / "avg_model" / "avg_model.pt"

# 加载预训练 backbone（跳过分类头）
model.load_pretrained(torch.load(pretrained, map_location='cpu', weights_only=True))
```

### 5.2 预训练权重加载策略

```python
def _load_backbone(model, state_dict, skip_keys=('projection', 'classifier')):
    # 1. 去掉 DDP 产生的 'module.' 前缀
    if all(k.startswith('module.') for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # 2. 跳过分类头（维度随 num_speakers 变化）
    filtered = {k: v for k, v in state_dict.items()
                if not any(k.startswith(s) for s in skip_keys)}
    # 3. strict=False — 允许部分键不匹配
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    logger.info("Loaded model: %.1fM params (clf excluded)", param_count)
```

### 5.3 优化器配置

```python
# 分组：backbone 低 lr, classifier 高 lr
backbone_params = [p for n, p in model.named_parameters()
                   if not n.startswith('projection.') and not n.startswith('classifier.')]
classifier_params = [p for n, p in model.named_parameters()
                     if n.startswith('projection.') or n.startswith('classifier.')]

optimizer = torch.optim.Adam([
    {'params': backbone_params, 'lr': 1e-4},        # backbone: 低学习率
    {'params': classifier_params, 'lr': 1e-3},       # 分类头: 10× lr
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

### 5.4 训练循环

```python
for epoch in range(epochs):
    model.train()
    for feats, labels in train_loader:
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(feats)                 # (B, num_speakers)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    scheduler.step()

    # 验证
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for feats, labels in val_loader:
            logits = model(feats.to(device))
            val_loss += nn.CrossEntropyLoss()(logits, labels.to(device)).item()
            val_acc += (logits.argmax(1) == labels.to(device)).sum().item()

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), OUTPUT_DIR / f"{model_name.lower()}_best.pt")
        torch.save(model.backbone_state_dict(), OUTPUT_DIR / f"{model_name.lower()}_backbone.pt")

torch.save(model.state_dict(), OUTPUT_DIR / f"{model_name.lower()}_final.pt")
```

### 5.5 保存文件

| 文件 | 内容 |
|------|------|
| `{model}_final.pt` | 最终完整权重的 state_dict（backbone + classifier） |
| `{model}_best.pt` | 验证集准确率最高的完整权重 |
| `{model}_backbone.pt` | 最佳模型去掉分类头后的 backbone 权重（用于部署推理） |

### 5.6 Trainer 封装

**代码位置：** `app/train/trainer.py`

`SpeakerTrainer` 类封装完整训练生命周期，支持按 Web 请求调用：

```python
class SpeakerTrainer:
    def run_training(self, model_name: str, epochs: int = 3,
                     batch_size: int = 64, base_lr: float = 1e-4) -> dict:
        """训练入口：
        1. 从 DB 加载训练数据
        2. 初始化模型 + 加载预训练权重
        3. 训练循环（进度报告通过回调）
        4. 保存 checkpoint + 注册到 DB
        5. 运行 ONNX 导出
        6. 返回训练指标
        """
```

---

## 6. 模型效果对比

**代码位置：** `app/train/evaluate.py`

### 6.1 评估指标

对微调前后模型进行 pairwise 比较：

| 指标 | 公式 | 说明 |
|------|------|------|
| 类内相似度 (Within) | `mean(cos(emb_i, emb_j))` for same speaker, i ≠ j | 同一说话人语音的相似度均值 |
| 类间相似度 (Between) | `mean(cos(emb_i, emb_j))` for different speakers | 不同说话人语音的相似度均值 |
| 分离度 (Separation) | `within.mean - between.mean` | 越大越好（说话人区分能力） |
| 改善率 | `count(improved) / N_spk` | 类内相似度提升的说话人占比 |

### 6.2 评估流程

```python
# 对每个评估模型:
for model_name, config in MODELS.items():
    # 预训练版本（baseline）
    embs_pretrain, labels = extract_embeddings(pretrained_ckpt, config, segments)

    # 微调版本
    embs_ft, _ = extract_embeddings(fine_tuned_ckpt, config, segments)

    # 计算统计
    within_pretrain, between_pretrain = compute_pairwise_stats(embs_pretrain, labels)
    within_ft, between_ft = compute_pairwise_stats(embs_ft, labels)

    # 逐说话人对比
    per_spk = {}
    for spk in unique_speakers:
        w_orig = mean(cosine_sim(pairs for spk))
        w_ft = mean(cosine_sim(pairs for spk))
        per_spk[spk] = {"orig": w_orig, "ft": w_ft, "delta": w_ft - w_orig}
```

### 6.3 结果持久化

评估结果写入 `model_versions` 表的 `metrics` JSON 列，格式：

```json
{
  "eval_type": "similarity_stats",
  "pretrained": {
    "within": {"mean": 0.653, "std": 0.089, "min": 0.412, "max": 0.891, "n": 48},
    "between": {"mean": 0.234, "std": 0.056, "min": 0.101, "max": 0.523, "n": 612}
  },
  "fine_tuned": {
    "within": {"mean": 0.721, "std": 0.072, "min": 0.503, "max": 0.934, "n": 48},
    "between": {"mean": 0.218, "std": 0.049, "min": 0.098, "max": 0.501, "n": 612}
  },
  "separation_delta": 0.068,
  "improved": 12,
  "degraded": 3,
  "per_speaker": {
    "SPK001": {"within_orig": 0.653, "within_ft": 0.721, "delta": 0.068},
    "SPK002": {"within_orig": 0.712, "within_ft": 0.689, "delta": -0.023},
    ...
  }
}
```

### 6.4 命令行评估

```bash
# 批量评估所有模型
python -m train.evaluate

# 指定模型和 DB
python -m train.evaluate --model CAM++ --db app/data/training.db

# 逐说话人详细输出
python -m train.evaluate --per-speaker

# 对比两个增量版本
python -m train.evaluate --compare v1 v2
```

---

## 7. 模型版本管理与部署

### 7.1 规范化目录结构

```
app/
  model_data/                  ← 统一模型存储根（MD）
    checkpoints/               ← PyTorch 快照
      CAM++/
        v0_pretrained/         ← 原始公开预训练（根节点）
          model.pt
          manifest.json
        v1/                    ← 第一轮增量训练
          backbone.pt
          manifest.json
        v2/                    ← 第二轮增量训练
          backbone.pt
          manifest.json
      ECAPA/
        v0_pretrained/
          model.pt
          manifest.json
        v1/
          backbone.pt
          manifest.json
      ResNet34/
        v0_pretrained/
          model.pt
          manifest.json
        v1/
          backbone.pt
          manifest.json
    deployed/                  ← ONNX 部署文件
      CAM++/
        v0.onnx
        v1.onnx
      ECAPA/
        v0.onnx
      ResNet34/
        v0.onnx
```

### 7.2 种子脚本

```bash
# 扫描现有 checkpoint 并注册到 model_versions + checkpoints 表
python scripts/seed_checkpoints.py
```

功能：
1. 扫描 `pytorch_weights/` 中的预训练和增量训练 checkpoint
2. 复制到 `model_data/checkpoints/{model}/{version}/` 规范化目录
3. 写入 `manifest.json`（含版本元数据、文件列表、父版本）
4. 注册到 `model_versions` 和 `checkpoints` 两张 DB 表
5. 记录演化链：`v0_pretrained → v1 → v2`

### 7.3 数据库表

**`model_versions`（部署版本记录）**

| 列 | 说明 |
|----|------|
| model_name | 模型名称 (CAM++/ECAPA/ResNet34) |
| version_tag | 版本标签 (v0_pretrained, v1, v2…) |
| base_model | 父版本 (如 `CAM++@v0_pretrained`) |
| embedding_dim | Embedding 维度 |
| score | 评估分数 |
| config | JSON 训练配置 |
| metrics | JSON 评估指标 |
| status | training/published/archived |
| created_at | 创建时间 |

**`checkpoints`（训练快照记录）**

| 列 | 说明 |
|----|------|
| model_name | 模型名称 |
| version_tag | 版本标签 |
| file_path | 快照文件路径 |
| embedding_dim | Embedding 维度 |
| metrics | JSON 评估指标 |
| is_published | 是否已发布 |
| created_at | 创建时间 |

### 7.4 部署流程

```
增量训练完成
  │
  ├─ ONNX 导出
  │   ├─ torch.onnx.export(model, dummy_input, output_path)
  │   └─ 动态轴：{0: "batch_size", 1: "time"}
  │
  ├─ 注册 model_versions 记录
  │   └─ insert_model_version(version=v2, base_model=v1, ...)
  │
  └─ 发布 ONNX
      ├─ 复制到 model_data/deployed/{model}/{version}.onnx
      └─ 更新 app/api/models/ 下的文件（热加载自动检测）
```

### 7.5 热加载机制

```python
class OnnxModel:
    """app/api/services/verifier.py"""
    # 每 30s 扫描 app/api/models/ 目录
    # ONNX 文件变更时自动重新加载 InferenceSession
    # 无需重启服务即可部署新模型
```

---

## 8. 命令行工具

### 8.1 预处理命令

```bash
# 待处理录音预处理
python -m train.preprocess

# 指定数量
python -m train.preprocess --limit 50

# 按业务系统筛选
python -m train.preprocess --limit 10 --biz collection

# 处理单条录音
python -m train.preprocess --call-id CALL001

# 持续监听
python -m train.preprocess --watch

# 不进行说话人标注（仅 VAD 切割）
python -m train.preprocess --no-diarize

# 指定声纹模型
python -m train.preprocess --diarizer-model ECAPA
```

### 8.2 增量训练命令

```bash
# Web 触发（通过模型管理页面）
# 或直接调用：
python -m train.fine_tune --model CAM++ --epochs 3

# 指定训练数据源
python -m train.fine_tune --db app/data/training.db

# 指定预训练权重
python -m train.fine_tune --pretrained pytorch_weights/campplus_cn_common.pt
```

### 8.3 评估对比命令

```bash
# 批量所有模型
python -m train.evaluate

# 指定模型
python -m train.evaluate --model CAM++

# 逐说话人详细输出
python -m train.evaluate --per-speaker
```

### 8.4 种子脚本

```bash
# 初始化模型 DB 记录（首次部署时运行）
python scripts/seed_checkpoints.py
```
