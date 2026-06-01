# ASV-Subtools Speaker Verification API

基于 FastAPI + ONNX Runtime 的声纹验证在线推理服务，面向银行客服和催收场景。

支持 **多模型**（CAM++ / ResNet34-LM / ECAPA-TDNN，运行时可切换）和
**多音频获取器**（本地文件 / S3 / Redis / MySQL，可插拔扩展）。

## 快速开始

### 环境准备

```bash
cd app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 下载模型

```bash
# 方案一：使用發布包（推荐）
# 从发布目录拷贝即可
cp release/asv-models/*.onnx api/models/
 
 # 方案二：单独下载各模型
 # CAM++（默认，中文场景推荐）
-mkdir -p models
-wget -O models/campplus.onnx \
+mkdir -p api/models
+wget -O api/models/campplus.onnx \
   https://modelscope.cn/models/damo/speech_campplus_sv_zh_en_16k-common_400k/resolve/master/onnx/model.onnx
 
 # ResNet34-LM（Wespeaker，256维）
-wget -O models/voxceleb_resnet34_LM.onnx \
+wget -O api/models/voxceleb_resnet34_LM.onnx \
   https://github.com/wenet-e2e/wespeaker/raw/master/pretrained_models/voxceleb_resnet34_LM.onnx
 
 # ECAPA-TDNN（VEDK00，192维）
-wget -O models/ecapa-speaker-v1.onnx \
+wget -O api/models/ecapa-speaker-v1.onnx \
   https://github.com/VEDK00/ecapa-speaker-v1/raw/main/model/ecapa-speaker-v1.onnx
```

### 启动服务

```bash
# 默认使用 CAM++ 模型，编辑 api/conf/config.yaml 中 model.path 可切换
cd api/
uvicorn main:app --host 0.0.0.0 --port 8000

# 生产环境多进程部署
cd api/
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

### 验证服务

```bash
curl -X POST http://localhost:8000/api/verify \
  -F "audio_a=@test_data/public/us_0010.wav" \
  -F "audio_b=@test_data/public/us_0011.wav"

# 预期响应：
# {"success":true,"is_same_speaker":false,"score":0.9691,...}
```

## ✅ 支持的模型

当前系统内置三个已测试验证的预训练模型，通过修改 `config.yaml` 中的
`model.path` 即可切换（无需重启服务—模型热更新会在 30 秒内自动检测并重新加载）。

| 模型 | 文件 | 维度 | 大小 | 交叉验证分数¹ | 同文件分数 | 推理耗时 | 特点 |
|------|------|------|------|-------------|-----------|---------|------|
| **CAM++** | `api/models/campplus.onnx` | 192 | 27MB | 0.9691 | 1.0000 | ~500ms | 达摩院出品，中文场景首选 |
| **ResNet34-LM** | `api/models/voxceleb_resnet34_LM.onnx` | 256 | 25MB | 0.9323 | 1.0000 | ~1.6s | Wespeaker，高分辨率 |
| **ECAPA-TDNN** | `api/models/ecapa-speaker-v1.onnx` | 192 | 80MB | 0.9986 | 1.0000 | ~1.7s | VEDK00 社区版 |

¹ 交叉验证分数使用 `test_data/public/us_0010.wav vs us_0011.wav`（非同一人）。

### 模型自动适配

系统在加载模型时自动检测 ONNX 图的输入/输出信息，无需手动配置 tensor 名称：

- **输入名自动检测**：优先使用 `feats`（CAM++），否则取第一个输入名
- **辅助输入自动补齐**：检测到 `feature_lens` / `input_lengths` 时自动传入实际帧数；
  检测到 `state` / `state_c` / `state_h` 等 RNN 状态时跳过（使用模型默认值）
- **数据类型自动转换**：根据 ONNX 元数据中的 `tensor(float)` / `tensor(int64)` 等
  信息自动匹配 numpy dtype
- **L2 归一化**：默认开启，可通过 `verification.normalize_embeddings` 配置

### 模型热更新

```yaml
# config.yaml（位于 api/conf/）
model:
  path: ./models/campplus.onnx      # 当前模型
  hot_reload_interval_sec: 30        # 每隔 30 秒检查文件变化
```

替换模型文件后，系统自动检测 md5 变化并重新加载，无需重启服务。

## API 接口

### `POST /api/verify` — 声纹验证

**模式A：直接上传音频文件 (multipart/form-data)**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| audio_a | File | 是 | 说话人A 音频文件 (WAV/ulaw/alaw/FLAC) |
| audio_b | File | 是 | 说话人B 音频文件 |
| scenario | str | 否 | 业务场景 (customer_service / debt_collection / audit) |
| threshold | float | 否 | 决策阈值覆盖 (0.0-1.0) |
| scoring_method | str | 否 | 评分方法 (cosine / euclidean / dot_product) |

**模式B：音频ID间接获取 (JSON body)**

请求示例：
```json
POST /api/verify/indirect
{
    "mode": "indirect",
    "audio_a": {
        "audio_id": "recording-20250101-001",
        "storage_backend": "nas"
    },
    "audio_b": {
        "audio_id": "recording-20250101-002",
        "storage_backend": "nas"
    },
    "scenario": "debt_collection",
    "threshold": 0.7
}
```

响应示例：
```json
{
    "success": true,
    "is_same_speaker": false,
    "score": 0.35,
    "threshold_used": 0.7,
    "processing_time_ms": 45.2,
    "embedding_a": {
        "dimension": 192,
        "source": "computed",
        "norm": 1.0
    },
    "embedding_b": {
        "dimension": 192,
        "source": "cached",
        "norm": 1.0
    },
    "scenario": "debt_collection"
}
```

### `GET /api/health` — 健康检查

## 配置管理

支持三种配置层级（优先级从高到低）：

1. **环境变量**: `ASV_MODEL_PATH`, `ASV_SERVER_PORT`, `ASV_VERIFICATION_DEFAULT_THRESHOLD` ...
2. **YAML 配置文件**: `config.yaml`（参见下文）
3. **内置默认值**

### 配置架构

主配置文件 (`config.yaml`) **只做选择，不做描述**：

| 模块 | 配置项 | 说明 |
|------|--------|------|
| **model** | `model.path` | 模型文件路径（通用参数如 provider/threads 也在此） |
| **fetcher** | `fetcher.type` | 仅指定类型 (local_file / s3 / redis / mysql)，无需写参数 |
| **audio** | `audio.*` | 音频处理通用参数（采样率、VAD、fbank） |
| **verification** | `verification.*` | 评分通用参数（阈值、评分方法、归一化） |

**模型特有参数**（如 ONNX 的输入/输出 tensor 名）由 Verifier 运行时从模型元数据自动检测，不在配置文件中指定。

**Fetcher 特有参数**（如 NAS 挂载路径、S3 密钥）由各 Fetcher 子类自身维护（硬编码默认值），不混入主配置。

### AudioFetcher — 可插拔音频获取（4种后端）

| 后端 | 类型名 | 适用场景 | 依赖 |
|------|-------|---------|------|
| **本地文件 / NAS** | `local_file` | 开发测试 / NAS 挂载 | 无 |
| **S3 兼容存储** | `s3` | 云端 / 对象存储 | `pip install boto3` |
| **Redis** | `redis` | 高速缓存 / 消息队列 | `pip install redis` |
| **MySQL** | `mysql` | 关系型数据库 | `pip install pymysql DBUtils` |

配置示例：

```yaml
# config.yaml（位于 api/conf/） — 仅需指定类型，无需任何后端参数
fetcher:
  type: local_file   # 切换为 s3 / redis / mysql 即可
```

**设计原则**：各后端的特有参数（NAS 挂载路径、S3 区域、Redis URL、MySQL 连接串）
由各 Fetcher 子类的 `__init__()` 硬编码管理，主配置文件不混入后端参数。
如需修改，直接在对应子类中编辑默认值即可。

扩展方式：

```python
# 如何新增一个存储后端：
from api.services.fetcher import AudioFetcher

@AudioFetcher.register("my_backend")
class MyFetcher(AudioFetcher):
    def __init__(self):
        self._my_setting = "/my/path"

    def fetch(self, audio_id: str, **kwargs) -> bytes:
        ...  # 你的实现
```

### 音频ID间接获取（Mode B）

当使用间接模式（`POST /api/verify/indirect`）时，请求体中的
`storage_backend` 字段告诉服务端使用哪个 Fetcher 根据 `audio_id` 获取音频。

```json
{
    "mode": "indirect",
    "audio_a": {
        "audio_id": "recording-20250101-001",
        "storage_backend": "nas"
    },
    "audio_b": {
        "audio_id": "recording-20250101-002",
        "storage_backend": "s3",
        "bucket": "my-asv-audio"
    },
    "scenario": "debt_collection",
    "threshold": 0.7
}
```

### 场景阈值

| 场景 | 阈值 | 说明 |
|------|------|------|
| customer_service | 0.45 | 客服场景：低门槛，避免误拒客户 |
| debt_collection | 0.70 | 催收场景：高门槛，避免误认 |
| audit | None | 审计场景：仅返回分数，不决策 |

## 项目结构

```
app/
├── api/                        # API 服务项目目录
│   ├── conf/
│   │   └── config.yaml         # 主配置文件
│   ├── models/
│   │   ├── campplus.onnx       # CAM++ 中文场景 192-dim (27MB)
│   │   ├── voxceleb_resnet34_LM.onnx  # ResNet34-LM 256-dim (25MB)
│   │   └── ecapa-speaker-v1.onnx  # ECAPA-TDNN 192-dim (80MB)
│   ├── __init__.py
│   ├── main.py                 # FastAPI 入口（生命周期管理）
│   ├── config.py               # YAML + 环境变量配置管理
│   ├── schemas.py              # Pydantic 请求/响应模型
│   ├── onnx_model.py           # ONNX 模型封装 + 热更新（~500ms 推理）
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py           # GET /health 健康检查（含模型/缓存状态）
│   │   └── verify.py           # POST /api/verify  &  /api/verify/indirect
│   └── services/
│       ├── __init__.py
│       ├── audio.py            # 音频加载/重采样/VAD/fbank（16kHz/8kHz 自适应）
│       ├── fetcher.py          # AudioFetcher 注册表 + 4 个后端实现
│       ├── cache.py            # 嵌入缓存（Redis / 内存两种后端）
│       └── verifier.py         # 核心验证（多模型自动适配 + 评分 + 决策）
├── sdk/                        # 多语言 SDK
│   ├── python/
│   │   ├── setup.py            # pip-installable
│   │   └── asv_sdk/__init__.py # ASVClient (verify_files / verify_ids)
│   ├── shell/
│   │   ├── asv_verify.sh       # curl 封装脚本
│   │   └── README.md
│   └── java/
│       ├── asv-sdk-*.jar       # Java SDK
│       └── README.md
├── test_data/public/           # 测试音频（us_0010.wav, us_0011.wav）
├── test/
│   ├── test_models.py           # API 端到端测试脚本
│   ├── test_sdk.py              # SDK 测试脚本（3 模型循环）
│   └── api_test_one.py          # 模型一键测试
├── requirements.txt
└── README.md
```

## 生产部署要点

### 模型选择建议

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 中文客服验证 | CAM++ (192-dim) | 达摩院中文预训练，速度快（~500ms），足够好的分离度 |
| 高精度要求（误判敏感） | ECAPA-TDNN (192-dim) | 交叉验证分数最高 (0.9986)，适合高风险催收场景 |
| 高维特征工程 | ResNet34-LM (256-dim) | 256 维输出，适合作为特征提取器用于下游任务 |
| 低延迟要求 (QPS > 2000) | CAM++ (192-dim) | 推理最快（~500ms），活跃 worker 可支撑高并发 |


### 嵌入缓存
- 高频客户 embedding 预存到 Redis，减少 ~50ms 推理时间
- 缓存 TTL 可配置（默认 24h）

### 多进程部署
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```
每个 worker 独立加载 ONNX 模型，ONNX Runtime C++ 内核释放 GIL，
多进程可实现 ~1500-3000 QPS（视硬件而定）。

### 8kHz 电话音频适配
CAM++ 等预训练模型原生工作在 16kHz。对于 8kHz 电话录音：
- **快速方案**: 上采样到 16kHz 后送模型（部分信息丢失）
- **推荐方案**: 用真实电话录音微调模型（需要带标注数据）

## 依赖

- Python >= 3.10
- ONNX Runtime >= 1.17
- FastAPI >= 0.110
- 详见 `requirements.txt`
