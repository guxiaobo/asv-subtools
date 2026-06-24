# 银行催收/客服场景声纹识别模型训练设计方案

> **2026-06-07 修订：** 新增实时录音推送 API、录音预处理 CLI 模块、模型增量训练模块，三者通过 SQLite 数据库串联为完整流水线。

---

## 1. 声纹样本量需求分析

### 1.1 关键假设

| 参数 | 催收场景 | 客服场景 |
|------|---------|---------|
| 坐席人数 | 200-500 人 | 500-2000 人 |
| 客户人数 | 10000-50000+ 人/月 | 10000-100000+ 人/月 |
| 单次通话时长 | 1-5 min | 3-15 min |
| 每人有效语音净时长 | 10-120 sec | 10-120 sec |

### 1.2 从零训练所需数据量

从零训练（non-pretrained）一个可用的声纹模型：

| 数据规模 | 最小可用 | 良好 | 工业级 |
|---------|---------|------|-------|
| 说话人数 | 1000 | 5000 | 10000+ |
| 每人录音数 | 5-8 条 | 8-15 条 | 15-30 条 |
| 每条时长 | 3-10 s | 5-30 s | 5-60 s |
| 总样本量 | ~8000 条 | ~60000 条 | ~300000 条 |
| 预期 EER | ~10% | ~5-7% | ~3-5% |

**结论：不建议从零训练。** 银行场景的实际数据量级（几百坐席 × 少量注册样本）完全不足以支撑从零训练。声纹模型学习的是"区分不同说话人"的通用能力，这需要大规模、多场景、多语种的数据。

### 1.3 推荐方案：预训练 + 微调

```
Stage 1: 大规模预训练基线
  ┌──────────────────────────────────────────┐
  │ 数据源: VoxCeleb1/2 (7k+ 说话人, 1M+ 条) │
  │        CN-Celeb (800+ 中国公众人物)        │
  │        VoxBlink (百万级说话人)             │
  │ 模型: ECAPA-TDNN / CAM++ / ResNet-221    │
  │ 产出: baseline checkpoint                │
  └──────────────────────────────────────────┘
            │
            ▼
Stage 2: 领域微调 (Domain Adaptation)
  ┌──────────────────────────────────────────┐
  │ 数据: 真实催收/客服录音 (已标记坐席ID)     │
  │ 规模: 500-2000 坐席 × 5-10 条 = 5000-2w  │
  │ 方式: 冻结 frontend, 微调 backbone+loss  │
  │ 产出: domain-adapted checkpoint          │
  └──────────────────────────────────────────┘
            │
            ▼
Stage 3: 小样本注册 (Enrollment)
  ┌──────────────────────────────────────────┐
  │ 每新坐席: 3-5 条录音注册, 提取 x-vector   │
  │ 客户: 通话中自动提取并缓存 embedding       │
  │ 阈值调优: 基于实际误识/拒识率校准          │
  └──────────────────────────────────────────┘
```

**微调阶段的具体样本需求：**

| 场景 | 最少 | 建议 | 说明 |
|------|------|------|------|
| 坐席 (agent) | 200 人 × 5 条 | 500+ 人 × 10 条 | 坐席 ID 有管理后台，标注成本低 |
| 客户混入 | 0 (不需要标注) | 1000+ 人 × 2-5 条 | 用于域内 channel 适配，非必须 |
| 注册 (新坐席) | 3 条 × 10s | 5 条 × 15s | 生产环境实际注册用，非微调用 |

### 1.4 银行场景的特殊性

| 挑战 | 影响 | 对策 |
|------|------|------|
| 录音通道分离 | 坐席/客户不同 MIC，信道差异大 | 微调时按通道打标，rank 后做 channel compensation |
| 双讲/重叠语音 | VAD 误切，质量下降 | 使用 voice activity detection 过滤双讲段 |
| 电话窄带 (8kHz) | 与 VoxCeleb (16kHz) 分布不同 | 降采样匹配，或 frontend 做重采样对齐 |
| 方言/口音 | 客户方多变 | 中文预训练模型 (CN-Celeb) 显著优于 VoxCeleb |
| 催收情绪波动大 | 发声方式偏离中性 | 收集高情绪样本加入微调 |

---

## 2. 录音接入与存储方案

### 2.1 系统架构总览

```
┌─────────────────────┐     ┌─────────────────────┐
│   催收业务系统 X     │     │  客服业务系统 Y      │
│  Debt Collection     │     │  Customer Service    │
│                     │     │                     │
│  ┌───────────────┐  │     │  ┌───────────────┐  │
│  │ 通话录音文件   │  │     │  │ 通话录音文件   │  │
│  │ (本地/远程)   │  │     │  │ (本地/远程)   │  │
│  └───────┬───────┘  │     │  └───────┬───────┘  │
│          │          │     │          │          │
│   POST /api/v1/recordings/push (REST API)        │
│   (坐席ID, 客户号码, 时间戳, 录音文件)           │
└──────────┼──────────┘     └──────────┼──────────┘
           │                           │
           ▼                           ▼
    ┌──────────────────────────────────────────┐
    │        ASV API 服务 (FastAPI)              │
    │                                            │
    │  ┌─────────────────────┐                   │
    │  │ POST /recordings/push│ ← 新增接口       │
    │  └──────┬─────┬───────┘                   │
    │         │     │                           │
    │         ▼     ▼                           │
    │  ┌────────┐ ┌───────────────┐             │
    │  │SQLite  │ │ 录音保存目录    │             │
    │  │数据库   │ │ local_recordings/│          │
    │  └────┬───┘ └───────────────┘             │
    └───────┼──────────────────────────────────┘
            │
            ▼  (CLI 模块依次消费)
    ┌──────────────────────────────┐
    │ 预处理模块 (python preprocess.py) │
    │  VAD → 通道拆分 → 噪音处理     │
    └───────────┬──────────────────┘
                │
                ▼
    ┌──────────────────────────────┐
    │ 增量训练模块 (python train.py)  │
    │  增量训练 → 评估 → 发布       │
    └───────────┬──────────────────┘
                │
                ▼  (模型发布)
    ┌──────────────────────┐
    │ api/models/ (ONNX)   │ ← ASV API 热加载
    └──────────────────────┘
```

### 2.2 实时录音推送 API（新增）

新增 REST API 接口，供催收和客服业务系统实时推送通话录音信息。

#### 2.2.1 接口定义

```
POST /api/v1/recordings/push
```

请求体（multipart/form-data 或 JSON）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `biz_system` | string | 是 | 业务系统标识：`collection`（催收）或 `cs`（客服） |
| `agent_id` | string | 是 | 坐席工号 ID |
| `customer_id` | string | 是 | 客户脱敏号码 |
| `call_timestamp` | string | 是 | 通话时间，ISO 8601 格式 |
| `call_id` | string | 是 | 通话唯一 ID（业务系统保证唯一） |
| `audio_source` | string | 是 | 录音来源类型：`binary` / `url` / `id` |
| `audio_data` | - | 条件 | 当 `audio_source=binary` 时：二进制音频文件（multipart upload） |
| `audio_url` | string | 条件 | 当 `audio_source=url` 时：可下载的录音文件 URL |
| `audio_id` | string | 条件 | 当 `audio_source=id` 时：业务系统内部录音 ID（用于 ID 映射查找） |
| `channel_separated` | bool | 否 | 是否已通道分离（双轨），默认 false |
| `duration_sec` | float | 否 | 通话总时长（秒），可选 |

响应：

```json
{
  "success": true,
  "data": {
    "recording_id": 10086,
    "call_id": "COL20260607_001",
    "local_path": "/data/local_recordings/collection/20260607/COL20260607_001_agent.wav",
    "status": "raw"
  }
}
```

#### 2.2.2 接口行为逻辑

```
收到请求
  │
  ├─ 1. 解析请求参数
  │    ├─ 校验必填字段
  │    └─ 校验 biz_system ∈ {collection, cs}
  │
  ├─ 2. 获取录音文件
  │    ├─ binary: 直接保存上传的二进制流
  │    ├─ url:    从 URL 下载到本地
  │    └─ id:     通过 fetcher 插件从业务系统拉取
  │
  ├─ 3. 保存录音文件到本地目录
  │    └─ {recordings_root}/{biz_system}/{date}/{call_id}.wav
  │
  ├─ 4. 写入 SQLite 录音记录表
  │
  └─ 5. 返回 recording_id
```

#### 2.2.3 录音保存目录结构

```
{recordings_root}/                    # 可在 config.yaml 中配置
├── collection/                       # 催收（按业务系统隔离）
│   ├── 20260607/
│   │   └── COL20260607_001.wav      # 原始录音
│   └── ...
└── cs/                               # 客服
    ├── 20260607/
    │   └── CS20260607_001.wav
    └── ...

{preprocessed_root}/                  # 预处理后输出
├── collection/
│   ├── 20260607/
│   │   ├── COL20260607_001_agent.wav
│   │   ├── COL20260607_001_customer.wav
│   │   └── COL20260607_001_segments/  # VAD 切段
│   └── ...
└── cs/
    └── ...
```

### 2.3 SQLite 数据库设计

整个流水线使用单一 SQLite 数据库文件（`{data_root}/training.db`），通过状态字段串联三个模块。

#### 2.3.1 录音记录表 (recordings)

```sql
CREATE TABLE recordings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- 元数据
    biz_system      TEXT    NOT NULL,          -- 'collection' | 'cs'
    call_id         TEXT    NOT NULL UNIQUE,   -- 业务系统通话 ID
    agent_id        TEXT    NOT NULL,          -- 坐席工号
    customer_id  TEXT    NOT NULL,          -- 客户脱敏号码
    call_timestamp  TEXT    NOT NULL,          -- ISO 8601
    channel_separated INTEGER DEFAULT 0,       -- 0/1
    duration_sec    REAL,
    -- 录音文件信息
    audio_source_type TEXT  NOT NULL,          -- 'binary' | 'url' | 'id'
    audio_original_url TEXT,                   -- 原始 URL（如有）
    local_audio_path    TEXT,                  -- 本地保存的原始录音路径
    -- 状态管理
    status          TEXT    NOT NULL DEFAULT 'raw',
        -- raw -> preprocessed -> trained
    pre_status      TEXT    DEFAULT 'pending',
        -- pending | processing | done | failed
    pre_result      TEXT,                      -- JSON: VAD 统计、质量分数等
    pre_error       TEXT,                      -- 错误信息（失败时）
    pre_finished_at TEXT,
    train_status    TEXT    DEFAULT 'pending',
        -- pending | processing | done | failed
    train_result    TEXT,                      -- JSON: 训练结果详情
    train_error     TEXT,
    train_finished_at TEXT,
    model_version   TEXT,                      -- 使用的模型版本
    -- 时间戳
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_recordings_status ON recordings(status);
CREATE INDEX idx_recordings_pre_status ON recordings(pre_status);
CREATE INDEX idx_recordings_train_status ON recordings(train_status);
CREATE INDEX idx_recordings_biz_agent ON recordings(biz_system, agent_id);
```

#### 2.3.2 模型版本表 (model_versions)

```sql
CREATE TABLE model_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    version         TEXT    NOT NULL UNIQUE,   -- 'v1.0', 'v1.1', ...
    -- 评估指标
    eval_metric     TEXT    NOT NULL,          -- 'EER' | 'minDCF'
    eval_value      REAL    NOT NULL,          -- 当前版本指标值
    prev_eval_value REAL,                      -- 上一版本指标值
    improved        INTEGER DEFAULT 0,         -- 0/1 是否优于上一版
    -- 训练信息
    train_recording_count INTEGER NOT NULL,    -- 训练使用的录音数
    train_speaker_count   INTEGER NOT NULL,    -- 训练使用的说话人数
    train_time_sec   REAL,                     -- 训练耗时
    previous_version TEXT,                     -- 上一版本号
    -- 模型文件
    model_path      TEXT    NOT NULL,          -- api/models/ 下的路径
    model_md5       TEXT,                      -- 文件 MD5
    is_active       INTEGER DEFAULT 0,         -- 当前是否生效版本
    notes           TEXT,                      -- 备注
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_model_versions_active ON model_versions(is_active);
```

#### 2.3.3 SQLite 配置项

SQLite 数据库路径及录音根目录在 `config.yaml` 中配置：

```yaml
# 新增配置段
training:
  db_path: "training.db"                # 相对路径（相对于 api/ 目录）或绝对路径
  recordings_root: "/data/local_recordings"
  preprocessed_root: "/data/preprocessed"
  # 预处理模块配置
  preprocessing:
    target_sample_rate: 16000
    min_segment_sec: 1.5
    max_segment_sec: 15.0
    snr_threshold: 15
    vad_window_ms: 30
    vad_threshold: 0.5
  # 增量训练配置
  incremental_train:
    base_lr: 0.0001
    epochs: 3
    batch_size: 64
    eval_test_set: "data/test"
    improvement_threshold: 0.001          # EER 提升至少达到此值才发布
```

### 2.4 坐席与客户标签策略

| 角色 | 标签来源 | 质量要求 | 用途 |
|------|---------|---------|------|
| 坐席 | 业务系统工号映射 | 高（系统保证唯一性） | 监督训练标签 |
| 客户 | 电话号码脱敏哈希 | 中（一人多号/多人一号时有错） | 非监督/UDA/辅助任务 |

**重要决策：坐席使用监督训练，客户建议使用无监督或自监督方法。**
- 客户可能借用手机、更换号码、多人共用
- 坐席 ID 是系统级唯一标识，可信度高

---

## 3. 录音预处理模块（新增）

### 3.1 模块定位

录音预处理模块是一个独立 CLI 工具，从 SQLite 数据库获取未处理的录音清单，循环进行训练前的录音处理。该模块**不依赖 ASV API 服务运行**，独立启动。

### 3.2 接口设计

```
# 命令行启动
python -m training.preprocess                     # 处理所有未处理的录音
python -m training.preprocess --limit 100         # 最多处理 100 条
python -m training.preprocess --biz collection    # 只处理催收录音
python -m training.preprocess --call-id COL...    # 处理单条
python -m training.preprocess --dry-run           # 仅检查清单，不处理
python -m training.preprocess --watch             # 持续监听模式：轮询新录音
```

### 3.3 处理流程

```
┌─────────────────────────────────────────────────────────┐
│  1. 读取 SQLite：SELECT * FROM recordings               │
│     WHERE status='raw' AND pre_status='pending'         │
│     按 created_at ASC 排序                             │
├─────────────────────────────────────────────────────────┤
│  2. 标记 pre_status = 'processing'                      │
│     写 updated_at                                       │
├─────────────────────────────────────────────────────────┤
│  3. 加载原始录音文件                                     │
│     └─ {recordings_root}/{biz_system}/{date}/{call_id}.wav │
├─────────────────────────────────────────────────────────┤
│  4. 录音预处理 Pipeline                                  │
│     ├─ ① 重采样至统一采样率 (8kHz/16kHz)               │
│     ├─ ② 通道拆分（如有双轨）                            │
│     ├─ ③ 能量检测 VAD 切段                               │
│     │    - 过滤前 2s（系统提示音）                       │
│     │    - 丢弃长度 < 1.5s 的段                          │
│     │    - 丢弃 SNR < 15dB 的段                          │
│     │    - 丢弃双讲段（坐席客户同时说话）                 │
│     ├─ ④ 噪音处理（降噪 / 能量归一化）                   │
│     └─ ⑤ 输出预处理文件                                  │
│         └─ {preprocessed_root}/{biz_system}/{date}/      │
│             ├── {call_id}_agent.wav                      │
│             ├── {call_id}_customer.wav                   │
│             └── segments/                                │
│                 ├── {call_id}_agent_seg001.wav           │
│                 ├── {call_id}_agent_seg002.wav           │
│                 └── ...                                  │
├─────────────────────────────────────────────────────────┤
│  5. 写预处理结果到 SQLite                                 │
│     ├─ status = 'preprocessed'（成功）                   │
│     ├─ pre_status = 'done' / 'failed'                    │
│     ├─ pre_result = JSON {                               │
│     │     segment_count: 12,                             │
│     │     agent_segments: 8,                             │
│     │     customer_segments: 4,                          │
│     │     avg_snr_db: 28.5,                              │
│     │     total_valid_sec: 64.2,                         │
│     │     dropped_segments: 3,                            │
│     │     dropped_reason: {"too_short": 2, "low_snr": 1} │
│     │   }                                                │
│     └─ pre_finished_at                                   │
└─────────────────────────────────────────────────────────┘
```

### 3.4 银行场景特有 VAD 规则

- 过滤前 2s（通常为系统提示音"本次通话将被录音"）
- 过滤长度 < 1.5s 的短段（语气词、嗯、啊）
- 双讲段强制丢弃（坐席与客户同时说话，声纹特征混叠）
- 信噪比阈值过滤：SNR < 15dB 的段丢弃

### 3.5 输出示例

```json
// pre_result JSON 示例
{
  "segment_count": 12,
  "agent_segments": 8,
  "customer_segments": 4,
  "agent_valid_sec": 45.3,
  "customer_valid_sec": 18.9,
  "avg_snr_db": 28.5,
  "min_snr_db": 16.2,
  "max_snr_db": 38.1,
  "dropped_segments": 3,
  "dropped_reason": {
    "too_short": 2,
    "low_snr": 1,
    "double_talk": 0
  }
}
```

---

## 4. 模型增量训练模块（新增）

### 4.1 模块定位

增量训练模块是一个独立的 CLI 工具，从 SQLite 数据库读取已完成预处理但尚未参与训练的录音及其预处理文件，执行增量训练，评估新模型，并根据效果决定是否发布。

### 4.2 接口设计

```
# 命令行启动
python -m training.incremental_train                    # 触发一轮增量训练
python -m training.incremental_train --dry-run           # 仅检查可用数据量
python -m training.incremental_train --force             # 即使无新增数据也重训
python -m training.incremental_train --epochs 5          # 覆盖默认 epoch 数
```

### 4.3 处理流程

```
┌──────────────────────────────────────────────────────────┐
│  1. 读取 SQLite                                           │
│     ├─ 待训练录音: SELECT * FROM recordings               │
│     │   WHERE status='preprocessed' AND train_status='pending' │
│     ├─ 当前生效模型: SELECT * FROM model_versions         │
│     │   WHERE is_active=1                                 │
│     └─ 检查是否有增量数据（无数据则直接退出）              │
├──────────────────────────────────────────────────────────┤
│  2. 组装增量训练集                                        │
│     ├─ 从 Kaldi 格式增量生成 wav.scp/utt2spk/spk2utt     │
│     └─ 说话人 head 扩展（新增 agent 自动扩展 softmax 权重） │
├──────────────────────────────────────────────────────────┤
│  3. 加载现有 checkpoint 进行增量训练                       │
│     ├─ 加载 base checkpoint（预训练模型或上一版本）       │
│     ├─ 扩展 classifier 层                                 │
│     │   old_weight = model.output_layer.weight [D×old_N] │
│     │   new_weight[D×new_N], 复制旧权重, XAVIER 初始化新增 │
│     ├─ 设置增量学习率 (base_lr × 0.05~0.1)               │
│     └─ 训练 1-3 epoch                                     │
├──────────────────────────────────────────────────────────┤
│  4. 模型评估                                              │
│     ├─ 在测试集上计算 EER / minDCF                        │
│     ├─ 与当前生效模型的指标对比                            │
│     └─ 判断是否 improvement                               │
│         new_eval < prev_eval - threshold  =>  improved    │
├──────────────────────────────────────────────────────────┤
│  5. 发布决策                                              │
│     ├─ 【Improved】                                        │
│     │   ├─ 导出 ONNX → {api_root}/models/{version}.onnx   │
│     │   ├─ 复制 latest.onnx → api/models/campplus.onnx   │
│     │   ├─ 写入 model_versions 表                          │
│     │   ├─ 将上一版本 is_active 置 0                       │
│     │   └─ 将新版本 is_active 置 1                         │
│     │                                                      │
│     ├─ 【Not Improved】                                     │
│     │   ├─ 记录失败原因                                    │
│     │   └─ 保留旧版本不变                                  │
│     │                                                      │
│     └─ 更新 recordings 表: train_status = 'done'/'failed'  │
│         train_result 写入训练细节                           │
└──────────────────────────────────────────────────────────┘
```

### 4.4 增量训练 vs 全量重训

| 维度 | 增量训练 (Incremental) | 全量重训 (Full Retrain) |
|------|----------------------|----------------------|
| 触发 | SQLite 有新增预处理数据 | 周度/半月度手动触发 |
| 耗时 | 10-30 min | 2-8 h |
| GPU 需求 | 1 卡 | 4-8 卡 |
| 数据偏移 | 容错低，需仔细调优 lr | 完全重新分布 |
| 遗忘风险 | ⚠️ 有 (catastrophic forgetting) | 无 |
| 推荐 | 日常迭代 | 定期基线刷新 |

**生产建议：日常增量 + 周度全量。**
- 周一至周五：增量训练模块（快速吸收新坐席、新数据分布）
- 周末或必要时：手动执行全量重训
- 全量训练完成后，将生成的 ONNX 模型替换至 api/models/

### 4.5 ONNX 导出与热更新

```
训练完成 (improved=true)
  │
  ├─▶ 检查/生成 api/models/ 目录
  │    └─ 目录已经存在（由 config.py _MODEL_DIR 自动创建）
  │
  ├─▶ 导出 ONNX：模型 → {version}.onnx
  │    └─ 同时复制为 campplus.onnx（ASV API 加载的文件名）
  │
  └─▶ ASV API 的 hot-reload 机制（30 秒轮询间隔）
       └─ 检测到文件 MD5 变更 → 自动重载模型
```

### 4.6 新说话人 head 扩展

声纹模型最后一层是 softmax / AM-Softmax，权重矩阵 `W ∈ R^{D×N}`（D=embedding dim, N=说话人数）。新增说话人时：

```python
# 核心逻辑
old_weight = model.output_layer.weight  # [D, old_N]
new_N = old_N + num_new_speakers
new_weight = torch.zeros(D, new_N)
new_weight[:, :old_N] = old_weight
# 新权重 XAVIER 初始化
nn.init.xavier_normal_(new_weight[:, old_N:])
model.output_layer.weight = nn.Parameter(new_weight)
```

---

## 5. 模块编排与数据流（SQLite 状态机）

### 5.1 状态流转

```
                          API 推送录音
                               │
                               ▼
                   status = 'raw'
                   pre_status = 'pending'
                               │
                       预处理模块读取
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
          pre_status = 'done'      pre_status = 'failed'
          status = 'preprocessed'  (记录错误，人工介入)
                  │
          增量训练模块读取
                  │
          ┌───────┴───────┐
          ▼               ▼
  train_status = 'done'  train_status = 'failed'
  (参与训练成功)          (记录错误)
          │
          ▼
  模型版本表新增记录
  (仅 improved=true 时)
```

### 5.2 模块启动方式

| 模块 | 启动方式 | 频次 |
|------|---------|------|
| ASV API（含录音推送接口） | `python api/main.py` 或 `uvicorn` | 持续运行 |
| 预处理模块 | `python -m training.preprocess [--watch]` | 手动或 cron 定时 |
| 增量训练模块 | `python -m training.incremental_train` | 手动或 cron（如每日凌晨） |

cron 示例：
```bash
# 每天凌晨 1 点执行预处理
0 1 * * * cd /app && python -m training.preprocess --limit 500

# 每天凌晨 3 点执行增量训练
0 3 * * * cd /app && python -m training.incremental_train
```

### 5.3 目录结构

```
{app_root}/
├── api/
│   ├── main.py                  # FastAPI 入口
│   ├── config.py                # 配置管理（含 training 配置段）
│   ├── routers/
│   │   ├── health.py
│   │   ├── verify.py
│   │   └── recordings.py        # 新增：录音推送 API 路由
│   ├── services/
│   │   ├── audio.py
│   │   ├── verifier.py
│   │   ├── fetcher.py
│   │   └── cache.py
│   ├── models/                   # ASV API 模型目录（写入此目录）
│   │   ├── campplus.onnx         # 当前生效模型（热加载）
│   │   ├── v1.0.onnx
│   │   └── v1.1.onnx
│   ├── docs/
│   │   └── voiceprint-training-design.md
│   └── conf/
│       └── config.yaml
│
├── training/                     # 新增：训练模块包
│   ├── __init__.py
│   ├── db.py                    # SQLite 数据库操作封装
│   ├── models.py                # SQLite 表模型定义
│   ├── schemas.py               # Pydantic 模型
│   ├── config.py                # 训练配置加载
│   ├── preprocess.py            # CLI: 预处理模块入口
│   ├── vad.py                   # VAD 处理实现
│   ├── audio_utils.py           # 音频工具函数（重采样、降噪等）
│   ├── incremental_train.py     # CLI: 增量训练模块入口
│   ├── trainer.py               # 增量训练核心逻辑
│   ├── evaluator.py             # 模型评估逻辑 (EER/minDCF)
│   ├── model_manager.py         # 模型版本管理（导出 ONNX、热更新）
│   └── utils.py                 # 通用工具
│
├── data/                        # 数据目录
│   ├── training.db              # SQLite 数据库
│   ├── local_recordings/        # 原始录音保存目录
│   │   ├── collection/
│   │   └── cs/
│   ├── preprocessed/            # 预处理后录音
│   │   ├── collection/
│   │   └── cs/
│   └── test_set/                # 固定测试集（用于模型评估）
└── ...
```

---

## 6. 生产关键风险（更新版）

### 6.1 灾难性遗忘 (Catastrophic Forgetting)

| 风险 | 表现 | 对策 |
|------|------|------|
| 增量学习率过大 | 旧说话人区分度下降 | lr = base_lr × 0.05~0.1 |
| 增量数据严重偏斜 | 新说话人过拟合 | 每轮增混 20~30% 历史代表性样本 |
| 长期不重训 | 质量阶梯式下降 | 强制执行周度全量重训 |
| **新增：** 发布不达标的模型 | 线上 EER 飙升 | **评估门槛：** 未超过上一版本时自动跳过发布 |

### 6.2 SQLite 并发风险

| 风险 | 影响 | 对策 |
|------|------|------|
| 多模块同时写 SQLite | 锁竞争 / 写冲突 | WAL 模式；每个模块独占自己负责的状态字段 |
| 预处理中服务重启 | 状态停留在 processing | 启动时检测 processing 状态标记回 pending |
| SQLite 数据库膨胀 | 查询性能下降 | 定期 VACUUM；按月归档历史数据 |

### 6.3 数据质量漂移

| 风险 | 表现 | 对策 |
|------|------|------|
| MIC 更换 | 信道特征突变，EER 飙升 | 录音元数据记录 MIC 型号；feature-level domain adaptation |
| 录音格式变更 | 采样率/编码变化 | 预处理集中统一重采样至 8kHz/16kHz |
| 坐席换人 | agent_id 与人不对应 | 系统审核机制，检测到 embedding 突变时告警 |

### 6.4 生产运维

| 风险 | 对策 |
|------|------|
| GPU 训练期间影响推理服务 | 训练与推理分 GPU 部署，或使用 MPS 分区 |
| ONNX 版本兼容 | api/models/ 保留前后版本；发布前在测试集验证 |
| 注册库与模型版本不一致 | 版本号关联，每次模型升级时重新提取所有注册 embedding |
| 录音下载超时/失败 | API 异步处理：先返回 recording_id，后台任务补下载 |

---

## 7. 配套工具建议

基于 ASV-Subtools 项目已有代码，建议开发以下配套工具：

| 工具 | 位置 | 功能 |
|------|------|------|
| `training/preprocess.py` | `app/training/` | 录音预处理 CLI（VAD、降噪、切段） |
| `training/incremental_train.py` | `app/training/` | 增量训练 CLI |
| `training/evaluator.py` | `app/training/` | 新模型评估，对比上一版本 |
| `training/model_manager.py` | `app/training/` | ONNX 导出、版本管理、热更新 |
| `training/vad.py` | `app/training/` | 银行场景特化 VAD 规则 |
| `tools/expand_classifier.py` | `pytorch/libs/nnet/` | 动态扩展 softmax 权重矩阵 |
| `tools/check_data_quality.py` | `pytorch/bin/` | VAD 质量统计、SNR 分布、异常检测 |
| `tools/eval_regression.py` | `score/` | 新旧模型回归测试，输出 EER diff |

---

## 8. 总结：你需要多少数据？

| 阶段 | 坐席数 | 每条录音时长 | 样本总量 | 方式 |
|------|-------|------------|---------|------|
| **POC 验证** | 50-100 | 5-10s × 5 条 | 250-500 | 直接用预训练模型 + 注册，不微调 |
| **试点上线** | 200-500 | 5-10s × 8 条 | 1600-5000 | 微调，期待 EER < 5% |
| **规模化运营** | 500-2000 | 5-30s × 10 条 | 5000-20000 | 持续增量训练，EER < 3% |
| **工业级** | 2000+ | 多样本覆盖场景 | 20000+ 累计 | 周度全量 + 日度增量 |

**核心结论：不需要从零训练。** 使用 VoxCeleb/CN-Celeb 预训练模型作为起点，只需要 500-2000 坐席 × 5-10 条真实录音做领域微调，即可达到可用水平 (EER 3-5%)。**本方案通过 SQLite + 三个模块（录音推送 API → 预处理 CLI → 增量训练 CLI）实现全流程自动化，每一步都有状态追踪和质量门槛。**
