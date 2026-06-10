# 增量训练优化方法论

> 生成时间: 2026-06-10
> 描述: 使用每日坐席录音进行 VAD 语音拆分、声纹相似度计算、说话人标记、
>       增量训练（fine-tune）和模型效果对比的完整流程。

---

## 目录

1. [整体流程图](#1-整体流程图)
2. [Phase 1: 录音预处理 (VAD 拆分)](#2-phase-1-录音预处理-vad-拆分)
3. [Phase 2: 说话人标注 (Diarization)](#3-phase-2-说话人标注-diarization)
4. [Phase 3: 跨录音客户声纹聚合](#4-phase-3-跨录音客户声纹聚合)
5. [Phase 4: 增量训练 (Fine-tune)](#5-phase-4-增量训练-fine-tune)
6. [Phase 5: 效果评估与模型发布](#6-phase-5-效果评估与模型发布)
7. [Phase 6: 模型对比实验 (增量实验验证)](#7-phase-6-模型对比实验-增量实验验证)
8. [完整命令速查](#8-完整命令速查)
9. [数据库 Schema](#9-数据库-schema)
10. [配置参考](#10-配置参考)
11. [常见问题与排查](#11-常见问题与排查)

---

## 1. 整体流程图

```
每日录音入库
    │
    ▼
┌─────────────────────────────┐
│ Phase 1: 录音预处理          │
│ • 降噪 (noise reduction)     │
│ • VAD 切割 (lossless)        │
│ • 保存段到 preprocessed/     │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Phase 2: 说话人标注 (Diarizer)│
│ • 提取每个段的 embedding     │
│ • 与坐席声纹对比 → Otsu 阈值   │
│ • 客户候选段 pairwise 聚类     │
│ • 输出: agent / customer      │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Phase 3: 跨录音聚合          │
│ • 按客户 ID 分组             │
│ • 跨录音客户段聚类            │
│ • centroid 写入 DB 声纹库     │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Phase 4: 增量训练 (Fine-tune)│
│ • 从 DB + preprocessed 读数据│
│ • 加载预训练 backbone        │
│ • 扩展分类器 (新说话人数)     │
│ • 训练 → 评估 → 导出          │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Phase 5: 效果评估            │
│ • 对比 fine-tune 前后         │
│ • EER / Separation 分析      │
│ • 模型发布 (ONNX 替换)       │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Phase 6: 增量实验验证        │
│ • 自适应阈值                │
│ • 新旧 enroll 对比           │
│ • PLDA 后端对比              │
│ • 跨模型 EER 排序            │
└─────────────────────────────┘
```

---

## 2. Phase 1: 录音预处理 (VAD 拆分)

### 2.1 入口

```bash
# 处理所有待处理录音
python -m train.preprocess

# 仅处理催收录音（前 50 条）
python -m train.preprocess --biz collection --limit 50

# 处理单条录音
python -m train.preprocess --call-id COL20220601-001

# 持续监听模式（60 秒轮询）
python -m train.preprocess --watch

# 跳过说话人标注（仅 VAD 切割）
python -m train.preprocess --no-diarize

# 使用指定模型做 diarization
python -m train.preprocess --diarizer-model ResNet34

# 仅执行跨录音聚合（跳过 VAD）
python -m train.preprocess --cross-aggregate-only

# 预览待处理清单（不实际执行）
python -m train.preprocess --dry-run
```

### 2.2 VAD 算法 (`app/train/vad.py`)

**核心函数**: `energy_vad()`

**参数**:

| 参数 | 默认值 | 说明 |
|---|---|---|
| `window_ms` | 30 | 帧长 (ms) |
| `threshold` | 0.5 | 噪声底噪之上的 dB 倍率 (实际 = max(threshold×10, 3) dB) |
| `lossless` | True | **关键参数**: True=保留所有段，不丢弃短段 |
| `filter_leading_sec` | 2.0 | 过滤开头系统提示音 |
| `max_segment_sec` | 15.0 | 超过此长度的段被切分 |

**算法步骤**:

1. **去头**: 丢弃前 `filter_leading_sec` 秒（过滤接通提示音/IVR）
2. **帧能量**: 50% 重叠 Hamming 窗 → 帧能量序列
3. **自适应阈值**: 噪声底噪 = P10 帧能量 + 至少 3dB 偏置
4. **获活检测**: `能量 > 阈值` → 语音帧
5. **段合并**: 间隙 < 0.8s 的连续段合并
6. **长段切分**: 超过 `max_segment_sec` 的段按最大长度切分
7. **lossless 模式**: 不丢弃短段、不按 SNR 过滤（默认开启）

**预处理管线**: `preprocess_recording()`

```
load_audio(audio_path) → resample to 16kHz
    ↓
reduce_noise(waveform)  # 可选降噪（谱减法）
    ↓
energy_vad(waveform, lossless=True)  # 语音段切割
    ↓
save_segments(output_dir, ...)  # 保存为 WAV 文件
    ↓
run_diarization()  # 说话人标注（默认开启）
```

### 2.3 输出目录结构

```
data/preprocessed/
  └── collection/                # 业务系统名
        └── 2022-04-20/          # 录音日期
              └── 田如兰-2204201033/   # call_id
                    ├── 田如兰-2204201033_seg000.wav
                    ├── 田如兰-2204201033_seg001.wav
                    └── ...
```

---

## 3. Phase 2: 说话人标注 (Diarization)

### 3.1 核心类: `SpeakerDiarizer`

**文件**: `app/train/diarizer.py`

```python
from train.diarizer import SpeakerDiarizer

diarizer = SpeakerDiarizer(
    model_path=None,           # None=自动使用 app/api/models/{model_name}.onnx
    model_name="CAM++",        # CAM++ / ResNet34 / ECAPA
    db_path=None,              # 数据库路径
    agent_threshold=None,      # None=自适应检测
    cluster_threshold=0.35,    # 客户聚类阈值
    min_samples=2,             # 最小聚类样本数
    adaptive_tuning=True,      # 自适应阈值调优
)

results = diarizer.diarize(segment_files)  # 标注所有段
summary = diarizer.summarize(results)      # 统计数据
```

### 3.2 算法流程（`diarize()` 方法）

```
输入: segment_files = [seg000.wav, seg001.wav, ...]
  │
  ├── 1. 提取每个段的 embedding (ONNX 推理)
  │     └── embedding = extract_embedding(wav)  # L2-normed vector
  │
  ├── 2. 与坐席参考声纹计算相似度
  │     ├── 从 DB speaker_voiceprints 读取坐席声纹 (model_name, agent)
  │     └── sim = dot(segment_emb, agent_ref_emb)
  │
  ├── 3. Otsu 自适应阈值分割
  │     ├── 收集所有有效相似度 [s1, s2, ..., sn]
  │     ├── 基于双峰分布 (Otsu's method) 寻找最佳分隔点
  │     ├── 得分 < 阈值 → 客户候选
  │     └── 得分 ≥ 阈值 → 坐席
  │
  ├── 4. 客户候选段 pairwise 聚类
  │     ├── 所有客户候选段之间计算相似度矩阵
  │     ├── adj = sim_mat >= cluster_threshold
  │     ├── connected_components → 找群组 (min_size=2)
  │     └── 最大群组 → 确认标注为 "customer"
  │
  └── 输出: [{idx, file, label, sim_to_agent, ...}, ...]
```

### 3.3 Otsu 自适应阈值（`_auto_detect_threshold`）

```python
def _auto_detect_threshold(self, scores):
    """基于双峰聚类寻找坐席-客户分界点。"""
    scores = np.sort(scores)
    best_t, best_bcv = scores[0], 0.0
    for t in scores:
        below = scores[scores <= t]
        above = scores[scores > t]
        if len(below) < 1 or len(above) < 1:
            continue
        # 类间方差 (Between-class variance)
        w1, w2 = len(below) / total, len(above) / total
        mu = np.mean(scores)
        mu1, mu2 = np.mean(below), np.mean(above)
        bcv = w1 * (mu1 - mu)**2 + w2 * (mu2 - mu)**2
        if bcv > best_bcv:
            best_bcv, best_t = bcv, t
    return best_t
```

### 3.4 客户聚类确认（`_diarize_otsu`）

```
客户候选段集合 = [段 with label == "customer_candidate"]

if len(valid_cust) < min_samples:
    → 全部直接标 customer（Otsu 分割本身就可靠）

elif 形成 ≥1 个群组:
    → 最大群组 → 标 customer + 计算 centroid
    → 其他群组 → 标 uncertain（多说话人场景）
    → 孤立段   → 标 uncertain

else:  # 无有效聚类但 Otsu 已分割
    → 全部标 customer
```

### 3.5 多说话人检测

在 `_diarize_legacy` 中（fallback 模式），当客户候选段聚类后出现以下任一情况触发多说话人检测：

1. **第二大群组 ≥ 主群组 60%** — 明显的第二客户
2. **群组数 ≥ 3 且主群组 ≤ 3 段** — 过于分散
3. **孤立段 ≥ 主群组 50% 且主群组 ≤ 3 段** — 大量无法归类的段

多说话人场景下，非主群组全部标 `uncertain`，不参与声纹提取。

### 3.6 客户声纹提取

```python
customer_vp = diarizer.get_customer_voiceprint(diar_results, segment_files)
# 返回: {"embedding": np.ndarray, "num_segments": int}
```

- 仅使用 label="customer" 的段
- 所有客户段 embedding 取均值 + L2 归一化
- 返回 centroid embedding

### 3.7 模型阈值速查

| 模型 | 聚类阈值 (cluster_threshold) | 备注 |
|---|---|---|
| CAM++ | 0.49 | 内存中默认 |
| ResNet34 | 0.59 | 内存中默认 |
| ECAPA | 0.68 | 内存中默认 |

---

## 4. Phase 3: 跨录音客户声纹聚合

### 4.1 触发时机

在预处理每批录音完成后自动触发（见 `app/train/preprocess.py` 的 `_cross_call_aggregate_phase`）。

### 4.2 聚合流程

```
跨录音聚合:
  1. 查询所有 pre_status='done' 的录音
  2. 按客户 ID 分组 (customer_phone 或 call_id 前缀)
  3. 每组内:
     a. 逐通录音跑一遍 diarizer (agent/customer 标注)
     b. 汇总所有客户段的 embedding
     c. 跨录音 channel 聚类 → centroid
  4. 注册到 speaker_voiceprints 表
```

### 4.3 核心 API

```python
diarizer.cross_call_aggregate(calls_by_customer)
# 返回: [{
#   "customer_id": str,
#   "embedding": np.ndarray,
#   "num_segments": int,
#   "num_calls": int,
#   "source_call_ids": [str, ...],
# }, ...]
```

---

## 5. Phase 4: 增量训练 (Fine-tune)

### 5.1 入口

```bash
# 训练所有三个模型（默认）
PYTHONPATH=. python app/train/fine_tune.py

# 训练指定模型
PYTHONPATH=. python app/train/fine_tune.py --model campplus
PYTHONPATH=. python app/train/fine_tune.py --model resnet
PYTHONPATH=. python app/train/fine_tune.py --model ecapa

# 自定义参数
PYTHONPATH=. python app/train/fine_tune.py \
  --epochs 20 \
  --lr 1e-4 \
  --batch-size 16 \
  --max-frames 400
```

### 5.2 核心函数: `train_model()`

```
train_model(model_name, epochs=5, lr=1e-4, batch_size=32, max_frames=400, val_split=0.15)
```

**步骤**:

1. **构建训练数据**: `build_training_data()` → 扫描 preprocessed 目录 + DB 映射
2. **加载模型**: 预训练权重 + 扩展分类器（新 num_speakers）
3. **数据分割**: 按说话人分层 85/15 划分
4. **训练**: Adam 优化器 + CosineAnnealingLR + CrossEntropyLoss
5. **验证**: 每个 epoch 后评估 val acc，保存最佳
6. **导出**: `{model}_best.pt`, `{model}_final.pt`, `{model}_backbone.pt`

### 5.3 关键参数配置 (`app/train/config.py`)

默认配置:
```yaml
incremental_train:
  base_lr: 0.0001       # backbone 学习率
  epochs: 3              # 默认 epoch 数
  batch_size: 64         # 批次大小
  improvement_threshold: 0.001  # EER 改进发布阈值
```

### 5.4 训练数据构建流程

```python
def build_training_data() -> (segments, id_to_speaker):
    """
    1. 从 DB 查询 recordings (status='preprocessed')
       → 获取 {call_id: customer_phone} 映射

    2. 扫描 preprocessed/collection/{date}/{call_id}/
       → 找到所有 VAD 段 WAV 文件

    3. 对每通录音（≥2 段）运行 diarizer 做说话人标注
       仅取 label="customer" 的段

    4. 按 customer_phone 映射到 speaker_id
       返回 [(wav_path, speaker_id), ...]
    """
```

### 5.5 说话人标签映射

- **坐席段**: 跳过（不用于训练） — `label != 'customer'` 的段被过滤
- **客户段**: 按 `customer_phone`（DB 字段）或 `call_id` 前缀命名映射为说话人
- **单段录音**: 无 agent/customer 对比时，全量作为客户段

### 5.6 优化策略

| 参数 | 值 | 原因 |
|---|---|---|
| backbone lr | 1e-4 | 低学习率微调，避免破坏预训练特征 |
| classifier lr | 1e-3 (10×) | 新分类器需要更快收敛 |
| weight_decay | 1e-4 | L2 正则化防过拟合 |
| gradient clip | 5.0 | 小数据集微调时防梯度爆炸 |
| scheduler | CosineAnnealing | 学习率平滑衰减 |

---

## 6. Phase 5: 效果评估与模型发布

### 6.1 评估脚本: `evaluate.py`

```bash
PYTHONPATH=. python app/train/evaluate.py \
  --model all \          # campplus/resnet/ecapa/all
  --epoch 20             # 训练 epoch 数（用于寻找对应 backbone 文件）
```

### 6.2 评估输出解读

```
Model: resnet                              ← 模型名称
  Segments: 125, Speakers: 15              ← 测试数据统计
  [Pretrained]  Within=0.6250  Between=0.2316  Separation=0.3934
  [Fine-tuned] Within=0.7485  Between=0.2212  Separation=0.5273
  [Delta]      Within Δ=+0.1235  Between Δ=-0.0104  Sep Δ=+0.1339

Per-speaker within-class similarity:
    张三           (n=12): orig=0.6234  ft=0.8012  Δ=+0.1778 ↑
    李四           (n= 8): orig=0.5987  ft=0.6543  Δ=+0.0556 ↑
    ...
  Summary: improved=8 degraded=3          ← 大多数说话人改善
```

### 6.3 关键评估指标

| 指标 | 含义 | 期望方向 |
|---|---|---|
| Within-class μ | 同人段间 cosine 相似度 | ↑ 越大越好 |
| Between-class μ | 异人段间 cosine 相似度 | ↓ 越小越好 |
| Separation | Within - Between | ↑ 越大越好（区分度） |
| EER | 等错误率 | ↓ 越小越好 |
| improved/degraded | 说话人级别改善计数 | 改善 > 退化 |

### 6.4 模型发布流程 (`app/train/model_manager.py`)

```bash
# 手动发布流程（incremental_train.py 自动执行）
1. ONNX 导出: pytorch/pipeline/export_onnx.py
2. 版本命名: v1.0 → v1.1 → ...
3. 复制到 api/models/{version}.onnx
4. 替换 api/models/campplus.onnx (热加载)
5. 注册 DB model_versions 表
```

---

## 7. Phase 6: 模型对比实验 (增量实验验证)

### 7.1 入口: `incremental_compare.py`

```bash
python -m app.train.incremental_compare
```

这是一个独立的对比实验脚本，用于验证增量训练的真实效果。

### 7.2 实验设计

**对比两个 enroll 方案**:

| 方案 | 说明 |
|---|---|
| **旧版 (DB)** | 使用 DB `speaker_voiceprints` 表中已有的声纹（可能是较早期的全局 centroid） |
| **新版 (centroid)** | 使用本次所有标注段的 centroid 作为 enroll（全量标注段计算） |

### 7.3 实验流程

```
1. 收集 segments: 扫描 preprocessed/collection/ 下所有段
   → [(speaker_name, wav_path), ...]

2. 对每个模型 (CAM++ / ResNet34 / ECAPA):

   a. 加载 ONNX 模型 → OnnxExtractor
   b. 从 DB 读取坐席声纹
   c. 提取所有段的 embedding + 与坐席的相似度
   d. 自适应阈值分割 (find_agent_threshold):
      计算得分分布中最大间隙点
   e. 标注段: sim >= threshold → agent, 否则 customer
   f. 构建新 enroll: 所有 agent 段 → centroid "000"
                        各 customer 段 → centroid {name}
   g. 旧 enroll: 从 DB speaker_voiceprints 读取
   h. 构建 test set: 每个说话人的所有段 embedding 列表
   i. 计算旧/新 EER:
      for each (enroll_vec, test_emb):
          score = dot(enroll_vec, test_emb)
          same = same_speaker ? 1 : 0
      compute_eer(all_scores, all_labels)
   j. (可选) PLDA 后端对比
   k. 输出对比结果表格
```

### 7.4 自适应阈值算法 (`find_agent_threshold`)

```python
def find_agent_threshold(all_scores):
    """
    从相似度分布中找最大间隙点作为坐席-客户分界。
    1. 排序 scores
    2. 计算相邻差分的滑动平均值
    3. 最大差分点 = 坐席-客户自然分界
    """
```

### 7.5 PLDA 后端

在 `incremental_compare.py` 末尾（数据足够时）自动执行 PLDA 对比：

- 基于测试集 embedding 训练 PLDA（15 次 EM 迭代）
- 使用训练后的 PLDA 模型对新旧 enroll 重新评分
- 输出 PLDA EER 和 minDCF

### 7.6 输出示例

```
  📊 CAM++ — 对比结果 (threshold=0.528)
  ══════════════════════════════════════════════════════════
  指标                    旧版 (DB)        新版 (centroid)
  ──────────────────────────────────────────────────────────
  Enroll 说话人           8                12
  Trials                  156              312
  同人/异人               72/84            144/168
  EER (%)                 6.372%           4.528%
  minDCF@0.01             0.0421           0.0310
  EER 变化                                    ↑1.844%

  得分分布 (cosine):
  旧版: 同人 μ=0.6234 σ=0.089  |  异人 μ=0.3456 σ=0.112  |  间隔=0.2778
  新版: 同人 μ=0.7123 σ=0.076  |  异人 μ=0.2987 σ=0.095  |  间隔=0.4136
```

### 7.7 EER 与 Separation 对比

| 对比项 | 用途 |
|---|---|
| **EER** | 端到端说话人验证精度（越小越好） |
| **Separation** | embedding 空间区分度（越大越好） |
| **PLDA 后端** | 验证线性变换能否进一步提升精度 |

---

## 8. 完整命令速查

### 8.1 数据准备

```bash
# 查看待处理录音
python -m train.preprocess --dry-run

# 全量预处理（含 diarization）
python -m train.preprocess

# 持续监听模式（60s 轮询）
python -m train.preprocess --watch

# 只做跨录音聚合
python -m train.preprocess --cross-aggregate-only
```

### 8.2 增量训练

```bash
# 训练三个模型（5 epoch）
PYTHONPATH=. python app/train/fine_tune.py

# 训练单个模型，更多 epoch
PYTHONPATH=. python app/train/fine_tune.py \
  --model resnet --epochs 20 --lr 1e-4 --batch-size 16
```

### 8.3 效果评估

```bash
# 对比 fine-tune 前后效果
PYTHONPATH=. python app/train/evaluate.py --model all --epoch 20

# 增量实验（新旧 enroll 对比）
python -m app.train.incremental_compare
```

### 8.4 后端处理（API 服务）

```bash
# 启动 ASV API（FastAPI）
uvicorn app.main:app --reload
```

---

## 9. 数据库 Schema

**文件**: `data/training.db`

### 9.1 recordings 表

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 ID |
| biz_system | TEXT | 业务系统 (collection/cs) |
| call_id | TEXT UNIQUE | 通话唯一标识 |
| agent_id | TEXT | 坐席工号 |
| customer_phone | TEXT | 客户电话号码（用作说话人标签） |
| call_timestamp | TEXT | 通话时间 |
| duration_sec | REAL | 录音时长 |
| local_audio_path | TEXT | 本地音频文件路径 |
| pre_status | TEXT | 预处理状态 (pending/processing/done/failed) |
| pre_result | TEXT | 预处理结果 JSON |
| train_status | TEXT | 训练状态 |
| status | TEXT | 总状态流转: raw → preprocessed → trained |

### 9.2 speaker_voiceprints 表

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INTEGER PK | 自增 ID |
| model_name | TEXT | 模型名 (CAM++/ResNet34/ECAPA) |
| speaker_type | TEXT | agent 或 customer |
| speaker_id | TEXT | 说话人ID (坐席="000", 客户=phone) |
| embedding | BLOB | float32 二进制，长度=嵌入维度×4 |
| segment_count | INTEGER | 计算时使用的段数 |
| source_call_ids | TEXT | JSON 数组，来源通话 ID 列表 |

### 9.3 model_versions 表

| 字段 | 类型 | 说明 |
|---|---|---|
| version | TEXT UNIQUE | 版本号 (v1.0, v1.1, ...) |
| eval_metric | TEXT | 评估指标 (EER) |
| eval_value | REAL | EER 值 |
| improved | INTEGER | 是否改进 |
| model_path | TEXT | ONNX 模型文件路径 |
| is_active | INTEGER | 是否当前生效 |

---

## 10. 配置参考

**文件**: `app/train/config.py` (默认值)

```yaml
# 完整配置项
db_path: "app/data/training.db"
preprocessed_root: "app/data/preprocessed"
recordings_root: "app/data/local_recordings"

preprocessing:
  target_sample_rate: 16000
  vad_window_ms: 30
  vad_threshold: 0.5          # 自适应底噪上的 dB 倍率
  min_segment_sec: 0.5        # lossless=False 时生效
  max_segment_sec: 15.0       # 长段切分阈值
  snr_threshold: 4.0          # lossless=False 时生效
  filter_leading_sec: 2.0     # 过滤开头系统音

incremental_train:
  base_lr: 0.0001             # backbone 学习率
  epochs: 3                   # 默认 epoch 数
  batch_size: 64              # 批次大小
  improvement_threshold: 0.001  # EER 改进发布阈值

model:
  backbone: "CAM++"           # 默认 backbone
  embedding_dim: 192          # 嵌入维度
```

---

## 11. 常见问题与排查

### 11.1 Otsu 阈值不稳定

- **现象**: 坐席与客户的得分分布无明显双峰（通话中仅一人说话时常见）
- **处理**: `diarizer.__init__(adaptive_tuning=True)` 自动回退到上一次有效阈值
- **监控**: 日志中 `Otsu 分割阈值=x.xxx` 超过 0.1-0.2 波动需关注

### 11.2 聚类阈值选择

- **CAM++ 0.49**: 语义空间较密，同类余弦相似度高，适合较低阈值
- **ResNet34 0.59**: 2D ResNet 空间较散，需要较高阈值
- **ECAPA 0.68**: 注意力池化后空间压缩，需要更高阈值
- 如果客户段被误标为 uncertain: `cluster_threshold` 调低
- 如果异人被误聚类: `cluster_threshold` 调高

### 11.3 训练失败（说话人种类不足）

- ```build_training_data``` 需要至少 2 个说话人才有意义
- 如果只有 1 个客户的所有录音，fine-tune 无法改善异人区分度
- 建议: 等积累 5+ 客户数据后再启动增量训练

### 11.4 fine-tune 后效果下降

- EER 反而上升 → 过拟合（epochs 过多 / 数据太少）
- 解决方法: 减小 lr (1e-5)、增加 weight_decay、减少 epochs
- Separation 下降 → backbone 被破坏（lr 太高）

### 11.5 DB 中无坐席声纹

- 首次部署需先录入坐席声纹
- 可使用 `diarizer.py` 中的 `get_customer_voiceprint` 类似方法提取坐席声纹
- 声纹写入 `speaker_voiceprints` 表，`speaker_type='agent'`, `speaker_id='000'`

### 11.6 跨录音聚合需要注意

- Phase 2 (跨录音聚合) 只处理 `pre_status='done'` 的录音
- 如果某个客户的录音日期分散在不同批次，需等所有录音处理完成后触发
- `--cross-aggregate-only` 可单独触发聚合（无需重跑 VAD）
