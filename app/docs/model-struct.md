# 模型架构与训练算法

> 更新时间：2026-06-22
> 代码位置：`app/train/fine_tune.py`（PyTorch 定义 + 训练循环）
> 推理引擎：`app/api/services/verifier.py`（ONNX Runtime）
> 评估脚本：`app/train/evaluate.py`
> 模型注册：`model_definitions` / `checkpoints` 表（SQLite）
> 打标轨迹：`segment_label_history` / `audio_segments.trained_status`

---

## 目录

1. [三模型概览](#1-三模型概览)
2. [CAM++ 架构详解](#2-cam-架构详解)
3. [ECAPA-TDNN 架构详解](#3-ecapa-tdnn-架构详解)
4. [ResNet34 架构详解](#4-resnet34-架构详解)
5. [训练算法](#5-训练算法)
6. [推理流程](#6-推理流程)
7. [评估方法](#7-评估方法)
8. [模型生命周期管理（SQLite 注册体系）](#8-模型生命周期管理sqlite-注册体系)
9. [打标生命周期与训练就绪状态](#9-打标生命周期与训练就绪状态)
10. [ONNX 部署映射](#10-onnx-部署映射)
11. [WEB API 架构总览](#11-web-api-架构总览)

---

## 1. 三模型概览

| 属性 | CAM++ | ECAPA-TDNN | ResNet34 |
|------|-------|------------|----------|
| Embedding 维度 | 192 | 192 | 256 |
| 参数估计 | ~6.2M | ~6.4M | ~8.3M |
| 输入特征 | 80-dim FBank | 80-dim FBank | 80-dim FBank |
| 预处理 | Mel 80 bins | Mel 80 bins | Mel 80 bins |
| 帧长/移 | 25ms / 10ms | 25ms / 10ms | 25ms / 10ms |
| 预训练权重 | `campplus_cn_common.pt` | `avg_model.pt` | `avg_model` |
| ONNX 部署 | `campplus.onnx` | `ecapa-speaker-v1.onnx` | `voxceleb_resnet34_LM.onnx` |
| 文件格式 | 2D → 1D 混合 | 1D Conv + Res2Net | 纯 2D ResNet |
| 池化层 | StatsPool (mean+std) | StatsPool (mean+std) | StatsPool (mean+std) |
| 年度性能 (VoxCeleb1-O EER) | ~0.8% | ~0.9% | ~1.1% |

---

## 2. CAM++ 架构详解

**代码位置：** `app/train/fine_tune.py:293–349`，类 `CAMPlus`

### 2.1 整体结构

```
Input: (B, T, 80)  ← FBank 特征（T帧 × 80 mel bins）
  │
  ├─ permute: (B, 80, T)   ← 转置为 (B, F, T) 形状
  │
  ├─ FCM（2D 前端卷积模块）
  │   └─ Conv2D(1→32, 3×3) → BN/ReLU
  │       └─ ResBlock(32, 2次, stride=2)  ← 频率轴 2x 降采样
  │       └─ ResBlock(32, 2次, stride=2)  ← 频率轴再 2x 降采样
  │       └─ Conv2D(32→32, 3×(2,1), stride=(2,1)) ← 频率轴再降采样
  │       └─ reshape → (B, 320, T)     ← out_channels = 32 * (80/8) = 320
  │
  ├─ xvector (nn.Sequential):
  │   │
  │   ├─ tdnn: TDNNBlock(320→128, k=5, stride=2)  ← 时间轴 stride=2 降采样
  │   │
  │   ├─ block1: CAMDenseTDNNBlock × 12  (k=3, d=1)
  │   │   └─ 密集连接: 每个子层输入 = concat(前面所有层输出)
  │   │   └─ growth_rate=32, bn_size=4 → bn_channels=128
  │   │   └─ Transit通道压缩: 128+12*32=512 → 256
  │   │
  │   ├─ block2: CAMDenseTDNNBlock × 24  (k=3, d=2)
  │   │   └─ 膨胀卷积 dilation=2 增大感受野
  │   │   └─ Transit通道压缩: 256+24*32=1024 → 512
  │   │
  │   ├─ block3: CAMDenseTDNNBlock × 16  (k=3, d=2)
  │   │   └─ Transit通道压缩: 512+16*32=1024 → 512
  │   │
  │   ├─ out_nonlinear: BN + ReLU
  │   ├─ stats: StatsPool → (B, 1024)   ← mean + std over time
  │   └─ dense: DenseLayer(1024→192)    ← 1×1 Conv1d + squeeze
  │
  └─ projection: Linear(192 → num_speakers)  ← 训练分类头（推理时移除）
```

### 2.2 CAMLayer（Context-Aware Masking）

核心创新 —— 可学习的时域注意力掩码：

```
输入 x: (B, C, T)
  │
  ├─ linear_local: Conv1d(C→out, k) → 局部时间卷积
  ├─ 分支: global_context = mean(x, dim=-1) + seg_pooling(x)
  │   └─ linear1(C→C/2) → ReLU
  │   └─ linear2(C/2→out) → Sigmoid → 注意力权重 m
  └─ 输出 = linear_local(x) * m
```

- `seg_pooling`: 分块池化（seg_len=100），将帧级特征分割为片段、池化、再上采样回原始分辨率
- 将全局上下文（整段 mean + 分段特征）通过一个小网络生成注意力权重，与局部卷积结果逐通道逐帧相乘
- 作用：抑制噪声帧、增强说话人特征明显的帧

### 2.3 Dense Block 细节

每个 `CAMDenseTDNNBlock` 内含 N 个 `CAMDenseTDNNLayer`：

```
Layer_i: BN/ReLU → Conv1x1(C_in→bn_channels) → BN/ReLU → CAMLayer(gal channels→out=32, k=[3])
```

每层输入维度 = 初始输入 + i×32（密集连接累加），输出 32 通道。

### 2.4 预训练加载

```python
model = CAMPlus(feat_dim=80, embedding_dim=192, num_speakers=N)
state = torch.load('campplus_cn_common.pt', map_location='cpu', weights_only=True)
# 去掉 'module.' 前缀（DDP）
# 跳过 'projection.' 和 'classifier.' 开头的键（维度随 num_speakers 变化）
model.load_pretrained(state_dict)  # strict=False
```

---

## 3. ECAPA-TDNN 架构详解

**代码位置：** `app/train/fine_tune.py:368–558`，类 `ECAPA_TDNNSpeaker`

### 3.1 整体结构

```
Input: (B, T, 80) → permute → (B, 80, T)
  │
  ├─ conv1: Conv1d(80→64, k=3, s=1) + BN/ReLU
  │
  ├─ block1: SE-Res2Block × 3
  │   └─ Conv1d(64→64, k=3, d=2) × 3 + Res2Conv + SE
  │
  ├─ block2: SE-Res2Block × 3
  │   └─ Conv1d(64→64, k=3, d=3) × 3 + Res2Conv + SE
  │
  ├─ block3: SE-Res2Block × 3
  │   └─ Conv1d(64→64, k=3, d=4) × 3 + Res2Conv + SE
  │
  ├─ conv2: Conv1d(64→192, k=1) ← 融合层，concat 三个 block 输出
  │   └─ concat: block1[−1] + block2[−1] + block3[−1] = 64×3 = 192
  │
  ├─ stats: StatsPool → (B, 384)  ← mean + std
  │
  ├─ bn1: BatchNorm1d(384)
  ├─ dense1: Linear(384→192) + BN1d
  ├─ dense2: Linear(192→192) + BN1d   ← embedding 输出
  │
  └─ projection: Linear(192 → num_speakers)  ← 分类头
```

### 3.2 SE-Res2Block 结构

```
输入 x: (B, C, T)
  │
  ├─ 1: BN/ReLU → Conv1d(C→C, k=1)  ← 1×1 降维
  ├─ 2: BN/ReLU → Res2NetConvBlock(C, k=3, d, scale=8)
  │   └─ channel 分割为 8 组（每组 C/8）
  │   └─ 第一组直通，后续每组 = Conv(前一组)+前一组输出
  │   └─ 类似层次化残差结构，增大感受野
  ├─ 3: BN/ReLU → Conv1d(C→C, k=1)  ← 1×1 恢复维度
  │
  ├─ SE-Block（Squeeze-Excitation）:
  │   └─ Global mean pooling → Linear(C→C/8) → ReLU
  │   └─ Linear(C/8→C) → Sigmoid → × input
  │
  ├─ 残差连接: 输出 = SE(output) + skip_connection(x)
  │   └─ skip_connection: 若通道匹配直接加，否则 Conv1d 匹配
```

### 3.3 关键参数

| 参数 | 取值 |
|------|------|
| Channels | 64（各 block 统一） |
| Res2Net scale | 8（每组 ~8 通道） |
| Dilation | block1:2, block2:3, block3:4 |
| SE reduction | 8 |
| Fusion | 三 block 末层 concat → Conv1d(192→192) |
| Embedding | 192（两个 dense layer） |
| AAM-Softmax | 否（此实现使用标准 CrossEntropy） |

---

## 4. ResNet34 架构详解

**代码位置：** `app/train/fine_tune.py:560–635`，类 `ResNet34_2D`

### 4.1 整体结构

```
Input: (B, T, 80)
  │   └─ unsqueeze(1).permute(0,1,3,2) → (B, 1, 80, T)
  │
  ├─ conv1: Conv2d(1→64, 3×3, s=1) + BN/ReLU
  │
  ├─ layer1: 3× BasicBlock2D(64→64,  s=1)
  ├─ layer2: 4× BasicBlock2D(64→128, s=2)  ← 频率轴 stride=2
  ├─ layer3: 6× BasicBlock2D(128→256, s=2) ← 频率轴 stride=2
  ├─ layer4: 3× BasicBlock2D(256→256, s=2) ← 频率轴 stride=2
  │
  ├─ reshape: (B, 256, H, T) → (B, 256*H, T)  ← 频率通道合并
  │   └─ H = ceil(80/8) = 10 → 2560
  │
  ├─ stats: StatsPool → (B, 5120)  ← mean + std over time
  │
  ├─ seg_1: Linear(5120→256)  ← embedding
  │
  └─ projection: Linear(256 → num_speakers)  ← 分类头
```

### 4.2 BasicBlock2D

```
输入 x: (B, C_in, F, T)
  │
  ├─ Conv2d(C_in→C_out, 3×(stride,1), pad=(1,0))  ← 频率维 stride、时间维 stride=1
  ├─ BN + ReLU
  ├─ Conv2d(C_out→C_out, 3×1, pad=(1,0))
  ├─ BN
  ├─ + shortcut（如需匹配维度: Conv2d 1×1 调整）
  └─ ReLU
```

频率维降采样 2 倍（layer2-4），时间维保持原长。总降采样倍率 = 2^3 = 8（3 个 stride=2 层）。

### 4.3 关键参数

| 参数 | 取值 |
|------|------|
| 频域分辨率 | 80 FBank → H=10 |
| 最终特征维度 | 256 × 10 = 2560 |
| StatsPool 输出 | 5120 (mean+std) |
| Embedding 维度 | 256 |
| Conv kernel | 3×3（时间×频率），频率维 stride 降采样 |

---

## 5. 训练算法

### 5.1 损失函数

```python
criterion = torch.nn.CrossEntropyLoss()
```

标准多分类交叉熵损失。将增量训练视为说话人分类任务，模型输出 `(B, num_speakers)` 与说话人标签 ID 计算交叉熵。

### 5.2 优化器：分组学习率 Adam

```python
optimizer = torch.optim.Adam([
    {'params': backbone_params, 'lr': lr},       # 骨干网络: 基础 lr
    {'params': classifier_params, 'lr': lr * 10}, # 分类头: 10× lr（快收敛）
], weight_decay=1e-4)
```

**原理：** 骨干网络已预训练，仅需微调（低学习率防止灾难性遗忘）；分类头（`projection`）是新初始化的，需要更高学习率快速收敛。

### 5.3 学习率调度器

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

余弦退火调度，`T_max=epochs`，学习率从初始值逐渐下降到接近 0。

### 5.4 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

全局梯度裁剪，范数上限 5.0，防止梯度爆炸（尤其在分类头 10× lr 时）。

### 5.5 训练循环（伪代码）

```
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        feats, labels → device
        optimizer.zero_grad()
        logits = model(feats)         # (B, num_speakers)
        loss = CrossEntropy(logits, labels)
        loss.backward()
        clip_grad_norm_(5.0)
        optimizer.step()
    scheduler.step()

    # 验证
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            logits = model(feats)
            loss, accuracy

    # 保存最佳模型
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'best.pt')
```

### 5.6 数据加载

```python
class SpeakerDataset(Dataset):
    def __getitem__(self, idx):
        wav_path, speaker_id = self.segments[idx]
        feat = self.fbank.extract_from_file(wav_path)  # (T, 80)
        # 中心裁剪 / 补零到 max_frames 帧
        if feat.size(0) > max_frames:
            start = (feat.size(0) - max_frames) // 2
            feat = feat[start:start + max_frames]
        else:
            pad = max_frames - feat.size(0)
            feat = F.pad(feat, (0, 0, 0, pad))
        return feat, speaker_id

collate_fn: stack + 转 tensor
DataLoader(batch_size=32, shuffle=True, drop_last=True, num_workers=0)
```

- `max_frames=400`：约 4 秒音频（400 帧 × 10ms 步进），中心裁剪
- `num_workers=0`：防止 macOS fork 问题
- `drop_last=True`：丢弃不完整 batch，防止 BN 层出现 batch_size=1

### 5.7 预训练权重加载与分类头替换

```python
# 示例：CAM++ 微调初始化
pretrained = torch.load('campplus_cn_common.pt', ...)
model = CAMPlus(feat_dim=80, embedding_dim=192, num_speakers=N_custom)
model.load_pretrained(pretrained)
# ⇒ 跳过 projection.weight/bias（维度不匹配），其余 backbone 权重复用
```

```python
def _load_backbone(model, state_dict, skip_keys=('projection','classifier')):
    if all(k.startswith('module.') for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    filtered = {k: v for k, v in state_dict.items()
                if not any(k.startswith(s) for s in skip_keys)}
    model.load_state_dict(filtered, strict=False)
```

---

## 6. 推理流程

### 6.1 ONNX 推理（生产部署）

**代码位置：** `app/api/services/verifier.py`

```
WAV 文件加载 (16kHz mono)
  │
  ├─ FBank 特征提取（torch-native 或 numpy 实现）
  │   └─ STFT(512, 160, 400) → Mel(80 bins) → log
  │   └─ 输出 shape: (T, 80)
  │
  ├─ ONNX Runtime 推理
  │   └─ 输入: (1, T, 80) float32
  │   └─ 输出: (1, 192) 或 (1, 256) ← embedding
  │
  ├─ L2 归一化
  │   └─ emb = emb / ||emb||₂
  │
  └─ 余弦相似度比较（验证场景）
      └─ score = dot(emb_A, emb_B)
      └─ decision = 'same' if score >= threshold else 'different'
```

### 6.2 模型输入特性

| 模型 | ONNX 输入名 | 预期形状 | 辅助输入 |
|------|------------|----------|---------|
| CAM++ | `feats` | `(1, T, 80)` | `feature_lens` (float32) |
| ECAPA | `input` | `(1, 1, 80, T)` | 无 |
| ResNet34 | `input.1` | `(1, 1, 80, T)` | 无 |

### 6.3 热加载机制

```python
# app/api/services/verifier.py
class OnnxModel:
    def __init__(self):
        self._sessions: Dict[str, ort.InferenceSession] = {}
        self._watcher: FileWatcher = ...
        # 每 30s 扫描 api/models/ 目录
        # 当 ONNX 文件变更时自动重新加载 session
```

---

## 7. 评估方法

**代码位置：** `app/train/evaluate.py`

### 7.1 评估指标

评估微调前后模型在客户说话人上的 embedding 区分度：

1. **类内相似度 (Within-class)** — 同一说话人两段 embedding 的余弦相似度均值
2. **类间相似度 (Between-class)** — 不同说话人两段 embedding 的余弦相似度均值
3. **分离度 (Separation)** — `within.mean - between.mean`，越大说明说话人区分越好
4. **改善/退化统计** — 逐说话人比较微调前后类内相似度变化

### 7.2 对比流程

```python
# 对每个模型:
model_orig = load_backbone(pretrained)          # 预训练版本
embs_orig, labels = extract_embeddings(segments) # 提取所有 embedding
stats_orig = compute_similarity_stats(embs_orig, labels)

model_ft = load_backbone(fine_tuned)             # 微调后版本
embs_ft, _ = extract_embeddings(segments)
stats_ft = compute_similarity_stats(embs_ft, labels)

# 逐说话人分析
for each speaker:
    orig_within = mean(cosine_sim(all pairs))
    ft_within = mean(cosine_sim(all pairs))
    delta = ft_within - orig_within
    print(f"{speaker}: {delta:+.4f} {'↑' if delta>0 else '↓'}")
```

### 7.3 评估结果持久化

评估结果写入 `model_versions` 表的 `metrics` JSON 列：

```json
{
  "eval_type": "similarity_stats",
  "pretrained": {"within": {mean,std,min,max,n}, "between": {...}},
  "fine_tuned": {"within": {...}, "between": {...}},
  "separation_delta": 0.0234,
  "improved": 8,
  "degraded": 3
}
```

---

## 附录 A：FBank 特征提取（纯 torch 实现）

**代码位置：** `app/train/fine_tune.py:642–710`，类 `FBankExtractor`

### 实现细节

```
输入波形 (float32, [-1,1], 16kHz)
  │
  ├─ STFT:
  │    n_fft=512, hop=160 (10ms), win=400 (25ms)
  │    hann 窗 → 功率谱
  │
  ├─ Mel 滤波器组:
  │    80 bins, 0–8000Hz
  │    三角滤波器（torch 原生实现，无 torchaudio 依赖）
  │
  ├─ Log: log(max(power, 1e-12))
  │
  └─ 输出: (T, 80) float32
```

特征提取完全基于 torch 原生操作（FFT 使用 `torch.stft`），不依赖 torchaudio/kaldi，确保在 ONNX 导出和边缘部署时的兼容性。

---

## 附录 B：模型参数量与预训练来源

| 模型 | 参数 | 预训练数据集 | 原始实现 |
|------|------|-------------|---------|
|| CAM++ | ~6.2M (backbone) | CN-Celeb | WeSpeaker / egrecho |
|| ECAPA-TDNN | ~6.4M | VoxCeleb1+2 | WeSpeaker / SpeechBrain |
|| ResNet34 | ~8.3M | VoxCeleb1+2 | ASV-Subtools v1 |

---

## 8. 模型生命周期管理（SQLite 注册体系）

**数据库表：** `model_definitions`、`checkpoints`、`checkpoint_training_segments`

### 8.1 模型架构定义 (`model_definitions`)

```
TABLE model_definitions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,              -- 模型名（如 "CAM++"）
    arch_version    TEXT    NOT NULL DEFAULT 'v1', -- 架构版本（结构性变更时递增）
    code_path       TEXT,                          -- 源码路径（如 pytorch/model/cam++.py）
    python_class    TEXT,                          -- Python 类名（如 "CAMPlus"）
    code_hash       TEXT,                          -- 源码指纹（非结构性变更时更新）
    embedding_dim   INTEGER NOT NULL DEFAULT 192,  -- embedding 维度
    feat_dim        INTEGER NOT NULL DEFAULT 80,   -- 输入特征维度
    num_speakers    INTEGER,                       -- 训练时说话人数量
    description     TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(name, arch_version)
)
```

**分支策略：**
- **结构性变更**（层数/维度改变导致 checkpoint 不兼容）→ 新建 `arch_version` 分支（如 `v1` → `v2`）
- **非结构性变更**（bugfix/超参调整）→ 同一 `arch_version` 内仅更新 `code_hash`
- 注册操作通过 `app/api/services/recording_db.py` 中的 `register_model_definition()` 完成

### 8.2 模型检查点 (`checkpoints`)

```
TABLE checkpoints (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    model_def_id        INTEGER REFERENCES model_definitions(id),
    model_name          TEXT    NOT NULL,
    version_tag         TEXT    NOT NULL,           -- 语义版本号（如 "v1.0", "v1.1"）
    base_checkpoint_id  INTEGER,                   -- 自引用：若为增量训练，指向 base checkpoint
    status              TEXT    NOT NULL DEFAULT 'created',
                                                    -- created | training | done | failed | published
    file_path           TEXT,                       -- ONNX/PyTorch 文件路径
    file_size           INTEGER,
    embedding_dim       INTEGER,
    metrics             TEXT,                       -- JSON：训练评估指标
    is_published        INTEGER NOT NULL DEFAULT 0, -- 是否发布为 API 服务使用
    trained_segments    INTEGER DEFAULT 0,          -- 训练使用的片段数
    trained_speakers    INTEGER DEFAULT 0,          -- 训练使用的说话人数
    description         TEXT,
    created_by          TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    UNIQUE(model_name, version_tag)
)
```

**DAG 版本链（Lineage）：**

每条 `checkpoint` 记录通过 `base_checkpoint_id` 自引用形成有向无环图：

```
pretrained (v1.0)
    ├── incremental_v1 (v1.1)  → 使用新打标片段增量训练
    │       └── incremental_v2 (v1.2)
    └── incremental_v2_direct (v1.1)  ← 另一条分支
```

- `base_checkpoint_id = NULL` 表示预训练版本（种子节点）
- `base_checkpoint_id` 指向父 checkpoint 的 `id`
- `status='published'` 且 `is_published=1` 表示当前 API 服务使用的版本

### 8.3 注册流程（种子脚本）

```bash
# 通过 WEB 界面操作
# 路径: /model-manager → 模型管理 → 注册新模型
#
# 或通过数据库直接操作:
python -c "
import aiosqlite, asyncio
async def register():
    db = await aiosqlite.connect('app/data/training.db')
    # 注册 CAM++ 定义
    await db.execute('''INSERT OR IGNORE INTO model_definitions
        (name, arch_version, code_path, python_class, code_hash, embedding_dim, feat_dim)
        VALUES (?, ?, ?, ?, ?, ?, ?)''',
        ('CAM++', 'v1', 'pytorch/model/cam++.py', 'CAMPlus',
         hashlib.sha256(open('pytorch/model/cam++.py','rb').read()).hexdigest()[:16],
         192, 80))
    # 创建初始 checkpoint
    await db.execute('''INSERT OR IGNORE INTO checkpoints
        (model_def_id, model_name, version_tag, status, embedding_dim, is_published)
        VALUES ((SELECT id FROM model_definitions WHERE name=?), ?, ?, 'done', 192, 1)''',
        ('CAM++', 'CAM++', 'v1.0'))
    await db.commit()
asyncio.run(register())
"
```

---

## 9. 打标生命周期与训练就绪状态

**数据库表：** `segment_label_history`、`audio_segments.trained_status`

### 9.1 片段标签状态机

```
                  ┌────────────────────────────┐
                  │  untrained (初始/默认状态)   │
                  │  — 已打标但未用于训练         │
                  └────────────┬───────────────┘
                               │
                  ┌────────────▼───────────────┐
            ┌────►│  training                    │
            │     │  — 正在被训练器消费          │
            │     └────────────┬───────────────┘
            │                  │
            │     ┌────────────▼───────────────┐
            │     │  trained                     │
            │     │  — 已被某次训练使用          │
            │     └────────────┬───────────────┘
            │                  │
            │     (用户重新打标)
            │                  │
            └─────────────────┘
               → 重置为 untrained
```

- `trained_status` 列在 `audio_segments` 表中
- 当用户**修改已打标片段**的标签时 → 自动重置为 `untrained`
- 训练器开始训练前 → 将消费的片段设为 `training`
- 训练成功完成后 → 将消费的片段设为 `trained`
- `count_labeled_segments_for_speakers()` 返回 `untrained` 计数，用于首页统计

### 9.2 打标历史日志

```sql
TABLE segment_label_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    segment_id      INTEGER NOT NULL REFERENCES audio_segments(id),
    old_label       TEXT    NOT NULL DEFAULT '',
    new_label       TEXT    NOT NULL DEFAULT '',
    old_speaker_type TEXT   NOT NULL DEFAULT '',
    new_speaker_type TEXT   NOT NULL DEFAULT '',
    old_ignored     INTEGER NOT NULL DEFAULT 0,
    new_ignored     INTEGER NOT NULL DEFAULT 0,
    operated_by     TEXT    NOT NULL DEFAULT 'admin',
    label_source    TEXT    NOT NULL DEFAULT 'manual',  -- manual | auto
    auto_reason     TEXT,           -- 自动打标的理由（相似度分数等）
    created_at      TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
)
```

示例日志记录：

| segment_id | old_label | new_label | label_source | auto_reason |
|-----------|-----------|-----------|-------------|-------------|
| 123 | "" | "AGENT_TELE" | manual | — |
| 123 | "AGENT_TELE" | "075582095333" | manual | — |
| 456 | "" | "@agent_755" | auto | "坐席 ID: 755, 声学特征匹配, 置信度: 0.85" |
| 456 | "@agent_755" | "__noise__" | manual | — |

### 9.3 训练器消费流程

```python
# 伪代码：增量训练中的片段消费
# 1. 查询未训练的片段
segments = db.get_untrained_segments_for_model(model_name)

# 2. 标记为 training
for seg in segments:
    db.update_trained_status(seg.id, 'training')

# 3. 执行训练
try:
    trainer.train(model, segments)
    # 标记为 trained
    for seg in segments:
        db.update_trained_status(seg.id, 'trained')
except Exception:
    for seg in segments:
        db.update_trained_status(seg.id, 'untrained')
```

---

## 10. ONNX 部署映射

### 10.1 PyTorch ↔ ONNX 版本对应

| 模型 | Embedding 维 | ONNX 文件名 | ONNX 输入名 | 输入形状 | 特征来源 | 输入归一化 |
|------|------------|------------|------------|----------|---------|-----------|
| CAM++ | 192 | `campplus.onnx` | `feats` | (1, T, 80) | FBank (torch) | log(x+eps) |
| ECAPA-TDNN | 192 | `ecapa-speaker-v1.onnx` | `input` | (1, 1, 80, T) | FBank (torch) | log(x+eps) |
| ResNet34 | 256 | `voxceleb_resnet34_LM.onnx` | `input.1` | (1, 1, 80, T) | FBank (torch) | log(x+eps) |

### 10.2 ONNX 导出流程

```
PyTorch checkpoint (.pt/.pth)
    │
    ├─ torch.onnx.export()
    │   └─ 动态 batch size
    │   └─ opset_version=15
    │   └─ input_names / output_names 固定
    │
    └─ saved to: app/api/models/{model_name}.onnx
```

### 10.3 部署切换

通过 WEB 界面 ("模型发布" 页面) 完成：

1. 用户选择某个 checkpoint → 点击 "发布"
2. 系统自动：
   - 将 checkpoint 对应的 ONNX 文件复制到 `api/models/` 目录
   - 更新 `checkpoints.is_published` 为 1（旧发布版本置为 0）
   - ONNX Runtime inference session 热加载（FileWatcher 每 30s 扫描）
3. 新版本立即生效，旧版本保留在磁盘上供回滚

### 10.4 热加载机制

```python
# app/api/services/verifier.py
class OnnxModel:
    def __init__(self):
        self._sessions: Dict[str, ort.InferenceSession] = {}
        # FileWatcher 每 30s 监控 api/models/*.onnx 的 mtime
        # 文件变更时自动重新创建 session，无中断切换
```

---

## 11. WEB API 架构总览

### 11.1 系统组件关系

```
┌──────────────────────── WEB 层 ────────────────────────┐
│  FastAPI + Jinja2 模板                                   │
│  ├─ app/api/routers/auth_router.py      登录/认证       │
│  ├─ app/api/routers/model_manager_router.py 模型管理员   │
│  │   ├─ 首页                          →  dashboard stats│
│  │   ├─ 录音断句 (segments)           →  VAD + 片段管理  │
│  │   ├─ 说话人打标 (label)            →  手动/自动打标   │
│  │   ├─ 增量训练 (train)             →  触发训练         │
│  │   └─ 模型管理 (model)            →  注册/发布         │
│  └─ app/api/routers/api_router.py    REST API           │
├────────────────────── 业务逻辑层 ──────────────────────┤
│  app/api/services/                                       │
│  ├─ recording_db.py    所有 SQLite CRUD 操作            │
│  ├─ verifier.py        ONNX Runtime 推理                │
│  └─ auth.py            认证/角色管理                    │
├────────────────────── 数据库层 ────────────────────────┤
│  app/data/database.py  SQLite schema + 迁移             │
│  app/data/training.db  运行时数据库                      │
├────────────────────── 训练层 ──────────────────────────┤
│  app/train/                                             │
│  ├─ fine_tune.py         模型定义 + 训练循环            │
│  ├─ trainer.py           增量训练调度                   │
│  ├─ preprocess.py        预处理流水线                   │
│  ├─ vad.py               VAD 断句                       │
│  ├─ diarizer.py          说话人聚类                     │
│  ├─ evaluate.py          评估                           │
│  ├─ model_manager.py     ONNX 导出 + 版本管理           │
│  ├─ evaluator.py         对比评估                       │
│  ├─ incremental_train.py 增量训练入口                   │
│  └─ db.py                训练用 SQLite CRUD             │
└─────────────────────────────────────────────────────────┘
```

### 11.2 数据流

```
录音上传 (通过 API 或 WEB)
    │
    ├─ 写入 training.db (recordings 表)
    ├─ 保存 WAV 到磁盘
    │
    ▼
VAD 断句 (WEB: /model-manager/segments)
    │
    ├─ 读取 WAV → energy_vad → 分片
    ├─ 写入 audio_segments 表 + WAV 片段存储
    │
    ▼
说话人打标 (WEB: /model-manager/label)
    │
    ├─ 手动: 下拉选已有说话人 / 输入新 ID / 设为噪音
    ├─ 自动: ONNX 模型推理 → 预览 → 确认
    └─ 写入 segment_label_history
    └─ trained_status = untrained
    │
    ▼
增量训练 (WEB: /model-manager/train)
    │
    ├─ 读取 untrained 片段
    ├─ 读取 base checkpoint
    ├─ 训练 → 新 checkpoint
    ├─ 评估 → 对比报告
    └─ 注册到 checkpoints 表
    │
    ▼
模型发布 (WEB: /model-manager/publish)
    │
    ├─ 导出 ONNX → api/models/
    ├─ 标记 is_published=1
    └─ verifier.py 热加载
```

---

## 附录 C：数据库表关系图

```
recordings
    │              ┌────────────────────────────────────┐
    ├─── 1:N ───► audio_segments                        │
    │               ├─ recording_id (FK → recordings.id)│
    │               ├─ segment_index (int)              │
    │               ├─ start_sec / end_sec / duration   │
    │               ├─ speaker_label                    │
    │               ├─ speaker_type                     │
    │               ├─ label_source (manual / auto)     │
    │               ├─ is_ignored (0/1)                 │
    │               ├─ trained_status                   │
    │               │   (untrained/training/trained)    │
    │               └─ batch_version                    │
    │                                                    │
    ├─── 1:N ───► segment_label_history                 │
    │               ├─ segment_id (FK)                  │
    │               ├─ old_label / new_label            │
    │               ├─ operator / label_source          │
    │               └─ auto_reason                      │
    │                                                    │
model_definitions                                        │
    │                                                    │
    └─── 1:N ───► checkpoints                           │
                    ├─ model_def_id (FK)                │
                    ├─ base_checkpoint_id (FK → self)   │
                    ├─ status / metrics / is_published  │
                    └─ file_path                        │
                         │                              │
                         └─── M:N ───► segment_label_history
                                       (checkpoint_training_segments 表)
```

---

*本文档最后更新：2026-06-22*
*对应标号 6 功能：首页统计 / Segments 页面 / 说话人打标页 / 增量训练 / 模型管理*
