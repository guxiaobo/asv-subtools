# 模型结构与训练算法

> 生成时间: 2026-06-10
> 源码位置: `app/pytorch_models/`（PyTorch 定义）和 `app/train/`（训练流程）

---

## 目录

1. [整体架构](#1-整体架构)
2. [CAM++ 模型](#2-cam-模型)
3. [ResNet34 模型](#3-resnet34-模型)
4. [ECAPA-TDNN 模型](#4-ecapa-tdnn-模型)
5. [公共组件 layers](#5-公共组件-layers)
6. [特征提取 (FBank)](#6-特征提取-fbank)
7. [训练算法细节](#7-训练算法细节)
8. [ONNX 推理模型](#8-onnx-推理模型)
9. [模型版本管理与发布流程](#9-模型版本管理与发布流程)
10. [模型对比 (evaluate.py)](#10-模型对比-evaluatepy)

---

## 1. 整体架构

系统维护 **三个声纹识别模型**，均采用预训练 backbone + 增量微调的方案：

| 模型 | 文件 | 嵌入维度 | 预训练来源 | 架构特点 |
|------|------|----------|------------|----------|
| **CAM++** | `campp_model.py` | 192 | 3D-Speaker (jingyaogong/campplus) | DenseNet + CAM (上下文感知掩码) |
| **ResNet34** | `resnet34_model.py` | 256 | WeSpeaker (avg_model) | 2D ResNet34 + 统计池化 |
| **ECAPA-TDNN** | `ecapa_model.py` | 192 | WeSpeaker (avg_model.pt) | SE-Res2Block + 注意力统计池化 |

**权重文件位置**: `app/pytorch_weights/`
- `campplus_cn_common.pt` — CAM++ 预训练权重 (9.5MB)
- `avg_model` — ResNet34 预训练权重 (目录，无后缀)
- `avg_model.pt` — ECAPA 预训练权重

**训练时创建的模型**（在 `app/pytorch_weights/fine_tuned/` 下）:
- `{model_name}_best.pt` — 最优 checkpoint
- `{model_name}_final.pt` — 最终 checkpoint（含 projection 分类器）
- `{model_name}_backbone.pt` — 去除分类器后的 backbone（用于部署推理）

---

## 2. CAM++ 模型

**文件**: `app/pytorch_models/campp_model.py`

### 2.1 CAM++ 论文与设计思想

CAM++ 的核心创新是 **Context Aware Masking (CAM)** 机制。传统 TDNN 使用固定的卷积核捕捉时间上下文，而 CAM 层动态地学习一个与输入相关的注意力掩码，使网络能够根据不同时段的语音特征自适应调整感受野。

### 2.2 整体结构

```
CamPP(input: B×T×80)
  ├── head: FCM (前端 CNN 模块)
  │     └── conv1(1→32) → BasicResBlock×2(stride=2) → BasicResBlock×2(stride=2)
  │     └── conv2(32→32, stride=(2,1)) → reshape → (B, 32×(80/8), T) = (B, 320, T)
  ├── xvector: Sequential
  │     ├── tdnn: TDNNBlock(320→128, k=5, stride=2)  # 初始降维
  │     ├── block1: CAMDenseTDNNBlock (12层, k=3, d=1)  # 第1密集块
  │     ├── transit1: TransitLayer  (降维 128+12*32 → /2)
  │     ├── block2: CAMDenseTDNNBlock (24层, k=3, d=2)  # 第2密集块 (膨胀卷积)
  │     ├── transit2: TransitLayer  (降维)
  │     ├── block3: CAMDenseTDNNBlock (16层, k=3, d=2)  # 第3密集块
  │     ├── transit3: TransitLayer  (降维)
  │     ├── out_nonlinear: BN+ReLU
  │     ├── stats: StatsPool (全局均值+标准差)
  │     └── dense: DenseLayer (通道数×2 → 192)
  └── output: (B, 192) embedding
```

### 2.3 关键子模块

#### FCM (Front-end CNN Module)
- 将原始的 80 维 fbank 特征通过 2D CNN 映射为时频特征
- 使用 `BasicResBlock`（2D 残差块，stride 仅在频率轴）
- 最终 reshape 为 (B, C×F', T) 供 1D TDNN 处理
- `out_channels = m_channels * (feat_dim // 8)` = 32 × 10 = **320**

#### CAMLayer (Context-Aware Masking)
- **双路径机制**:
  - **局部路径**: `Conv1d` + 常规 1D 卷积提取局部特征
  - **上下文路径**: 全局均值池化 + 分段池化 → 两层 FC → Sigmoid → 注意力掩码
- 最终输出 = `y * m`（局部特征 × 上下文注意力掩码）
- `seg_pooling`: 将时间轴分段池化 (seg_len=100)，上采样回原长度，提供分段级别的上下文

#### CAMDenseTDNNBlock
- 密集连接 (DenseNet 风格)：每层输入 = 之前所有层的输出拼接
- 每层 = BN+ReLU → 1×1 Conv (bottleneck) → BN+ReLU → CAMLayer
- 增长率 (growth_rate) = 32：每层输出通道数

#### TransitLayer
- 密集块间的过渡层：BN+ReLU → 1×1 Conv (降维 2×)
- 压缩信息密度，控制参数增长

### 2.4 前向传播维度变化

```
输入:  (B, T, 80)          特征维度 80
permute → (B, 80, T)
head:  (B, 320, T)         FCM 输出
tdnn:  (B, 128, T/2)       stride=2 下采样
block1: (B, 128+12*32, T/2) = (B, 512, T/2)   12 层密集增长
transit1: (B, 256, T/2)    降维
block2: (B, 256+24*32, T/2) = (B, 1024, T/2)  24 层密集增长
transit2: (B, 512, T/2)    降维
block3: (B, 512+16*32, T/2) = (B, 1024, T/2)  16 层密集增长
transit3: (B, 512, T/2)    降维
stats:  (B, 1024)          均值+标准差拼接
dense:  (B, 192)           最终 embedding
```

---

## 3. ResNet34 模型

**文件**: `app/pytorch_models/resnet34_model.py`

### 3.1 设计思想

标准 ResNet34 架构，适用于说话人识别。相比图像 ResNet，去掉了最后的全局平均池化步长，改为直接 `AdaptiveAvgPool2d(1,1)`。后端用 `StatsPool` + `Linear` 输出 embedding。

系统中有两个版本的 ResNet34：
1. **`ResNet34`** (campp_model.py 下的纯 PyTorch 版本) — 较传统，2D 卷积，适应 WeSpeaker 检查点
2. **`ResNet34_2D`** (fine_tune.py 内的版本) — 更灵活，支持动态 feat_dim，测试用的训练变体

### 3.2 结构

```
ResNet34(input: B×T×80)
  ├── conv1: Conv2d(1→32, 3×3, stride=1) + BN + ReLU
  ├── layer1: BasicBlock ×3 (32→32, stride=1)
  ├── layer2: BasicBlock ×4 (32→64, stride=2)
  ├── layer3: BasicBlock ×6 (64→128, stride=2)
  ├── layer4: BasicBlock ×3 (128→256, stride=2)
  ├── avgpool: AdaptiveAvgPool2d(1,1) → flatten → (B, 256)
  └── fc: Linear(256→256) → embedding
```

### 3.3 BasicBlock

标准残差块：
```
conv3×3 → BN → ReLU → conv3×3 → BN → + shortcut → ReLU
```
`expansion=1`，输出通道 = 输入通道数。

### 3.4 与 fine_tune.py 中 ResNet34_2D 的区别

`fine_tune.py` 中的 `ResNet34_2D`:
- `StatsPool` 替换 `AdaptiveAvgPool2d`：输出均值+标准差拼接
- 支持 `feat_dim` 可配置（通过 `h_freq = ceil(feat_dim/8)` 计算）
- 支持 `embedding_dim` 可配置
- 与训练流程集成：选择框 `return_embedding` 控制是否返回 embedding 或分类 logits

---

## 4. ECAPA-TDNN 模型

**文件**: `app/pytorch_models/ecapa_model.py`

### 4.1 设计思想

ECAPA-TDNN 是 2020 年提出的增强版 TDNN，核心创新：
1. **SE-Res2Block**: Res2Net 的多尺度卷积 + Squeeze-and-Excitation 通道注意力
2. **多尺度特征融合**: 3个不同膨胀率的 SE-Res2Block 输出拼接
3. **注意力统计池化 (Attentive StatsPool)**: 时间维度的注意力加权均值/标准差

### 4.2 结构

```
EcapaModel(input: B×T×80)
  ├── ecapa: EcapaTdnn (核心骨干)
  │     ├── layer1: Conv1d(80→512, k=5) + BN + ReLU
  │     ├── layer2[0]: SE_Res2Block(512, dilation=1)   # 近端上下文
  │     ├── layer2[1]: SE_Res2Block(512, dilation=2)   # 中程上下文
  │     ├── layer2[2]: SE_Res2Block(512, dilation=3)   # 远程上下文
  │     ├── concat: 拼接三层输出 → (B, 512*3, T)
  │     ├── conv: Conv1d(1536→1536, k=1) + BN + ReLU  # MFA 层
  │     └── output: (B, 1536, T)
  ├── attention: Attentive StatsPool (torch-native)
  │     ├── linear1: Conv1d(1536→128, 1) + Tanh
  │     ├── linear2: Conv1d(128→1536, 1) + Softmax (时间注意力)
  │     ├── mean = Σ(α·x)           # 注意力加权均值
  │     └── std = sqrt(Σ(α·x²) − mean²)  # 注意力加权标准差
  ├── bn: BatchNorm1d(3072)          # 拼接后归一化
  ├── fc: Linear(3072→192)           # embedding 层
  └── bn_out: BatchNorm1d(192)       # 输出归一化
```

### 4.3 SE-Res2Block

```
输入 x (B, C, T)
  ├── conv1×1 → BN → ReLU                              # 降维/升维准备
  ├── Channel Split: 分成 scale=8 组 (每组 C/8 通道)
  │     └── 每组: Conv1d(C/8, C/8, k=3, dilation=d) → BN → ReLU → BN → ReLU
  │            （第2组的输入包含第1组的输出 — 层次化特征）
  ├── concat groups → (B, C, T)
  ├── conv1×1 → BN → ReLU                              # 合并
  ├── SE Module:
  │     ├── GlobalAvgPool → Conv1d(C→C/8) → ReLU → Conv1d(C/8→C) → Sigmoid
  │     └── x = x · attention                          # 通道重标定
  └── + residual → 输出
```

### 4.4 前向传播维度变化

```
输入:   (B, T, 80)
permute → (B, 80, T)
layer1: (B, 512, T)                   初始映射
layer2[0]: (B, 512, T)                膨胀=1
layer2[1]: (B, 512, T)                膨胀=2
layer2[2]: (B, 512, T)                膨胀=3
concat:  (B, 1536, T)                 三尺度拼接
conv:    (B, 1536, T)                 MFA 融合
attention: (B, 3072)                  注意力加权 mean+std
bn:      (B, 3072)
fc:      (B, 192)                     最终 embedding
bn_out:  (B, 192)                     输出归一化
```

### 4.5 预训练权重的键名映射

Wespeaker 的 checkpoint 键名与自定义 EcapaModel 不同，`create_ecapa_model` 中包含完整的键映射逻辑：

| Wespeaker 键 | 本模型键 | 说明 |
|---|---|---|
| `layer1.conv` | `ecapa.layer1.conv` | 首层卷积 |
| `layer2.se_res2block.N.0-4` | `ecapa.layer2.N.0-4` | SE-Res2Block 子层 |
| `conv` | `ecapa.conv` | MFA 卷积 |
| `bn` | `ecapa.bn` | MFA BN |
| `attention.linear1/2` | `attention.linear1/2` | 注意力层（直接匹配） |
| `bn2` | `bn` | 统计池化后的 BN |
| `fc` | `fc` | 全连接层（直接匹配） |
| `bn3` | `bn_out` | 输出 BN |

---

## 5. 公共组件 Layers

**文件**: `app/pytorch_models/components.py`

公共组件库，被三个模型共享：

| 组件 | 用途 |
|---|---|
| `Mish` | Mish 激活函数 `x * tanh(softplus(x))` |
| `Swish` | Swish 激活函数 `x * sigmoid(x)` |
| `Nonlinearity(nl)` | 工厂函数：返回 ReLU / Mish / Swish / SiLU / Identity |
| `get_bn_relu(channels)` | BN + ReLU 顺序序列 |
| `statistics_pooling(x)` | 全局均值+标准差池化（functional） |
| `StatsPool` | `statistics_pooling` 的 Module 封装 |
| `TDNNBlock` | 1D 卷积 + 非线性 + BN（支持 pre_norm 顺序切换） |
| `DenseLayer` | 继承 TDNNBlock，1×1 Conv + BN (affine=False) 用于最终 embedding |
| `SERes2Block` | SE-Res2Block 的独立实现（与 ecapa_model 中的不同版本） |

---

## 6. 特征提取 (FBank)

**文件**: `app/train/fine_tune.py` 第 642-703 行 (FBankExtractor 类)

纯 PyTorch/numpy 实现，无 torchaudio 依赖：

| 参数 | 默认值 | 说明 |
|---|---|---|
| n_mels | 80 | Mel 滤波器组数 |
| sr | 16000 | 采样率 |
| n_fft | 512 | FFT 点数 |
| hop_length | 160 | 帧移 (10ms @ 16kHz) |
| win_length | 400 | 帧长 (25ms @ 16kHz) |
| f_min | 0 | 最低频率 |
| f_max | 8000 | 最高频率 |

**流程**: `波形 → STFT → Mel滤波器组 → log → (T, 80)`

---

## 7. 训练算法细节

**文件**: `app/train/fine_tune.py` 函数 `train_model()`

### 7.1 数据加载

```python
build_training_data()  →  [(wav_path, speaker_id), ...], {id: name}
```

- 从 `data/training.db` 查询 `recordings` 表（`status='preprocessed'`）
- 从 `data/preprocessed/collection/{date}/{call_id}/` 读取 VAD 段
- 每通录音运行 `SpeakerDiarizer.diarize()` 区分坐席/客户段
- **仅取客户段**用于训练（坐席段跳过）
- 按说话人（customer_phone）映射为 speaker_id
- 说话人均衡：每条录音的 diarizer 输出中，客户段被收集到对应说话人名下

### 7.2 数据集分割

每个说话人在各自段中按 85/15 随机划分（`val_split=0.15`）：
- 使用 `defaultdict` 收集每个说话人的段
- 每个说话人至少保留 1 个验证段（`n_val = max(1, int(n * val_split))`）

### 7.3 SpeakerDataset 与 DataLoader

```python
class SpeakerDataset(Dataset):
    def __getitem__(self, idx):
        wav_path, sid = self.segments[idx]
        feat = fbank.extract_from_file(wav_path)   # (T, 80)
        if T > max_frames:   随机裁剪一段
        elif T < max_frames: 零填充到 max_frames
        return feat, sid
```

- `max_frames=400`（对应 4s @ 10ms 帧移）
- `collate_fn`: 堆叠 feat 和 sid 为 batch
- `batch_size=32`（默认），`shuffle=True`
- `drop_last=True`（防止最后一个 batch shape 不一致）
- `num_workers=0`（避免多进程 init 问题）

### 7.4 模型加载与分类器扩展

```python
model = CAMPlus(feat_dim=80, embedding_dim=192, num_speakers=num_speakers)
state_dict = torch.load(ckpt_path, weights_only=True)
model.load_pretrained(state_dict)
```

关键：**预训练权重不含分类器**（`num_speakers` 为 None 时不创建 `projection` 层），训练时才传入实际说话人数以创建新分类器。

`load_pretrained` 方法使用 `strict=False` 加载，仅过滤掉 `projection.`/`classifier.` 前缀的键名。

### 7.5 优化器与学习率

```python
backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n and 'projection' not in n]
classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n or 'projection' in n]

optimizer = torch.optim.Adam([
    {'params': backbone_params, 'lr': lr},        # 1e-4
    {'params': classifier_params, 'lr': lr * 10}, # 1e-3 — 新分类器用更高 lr
], weight_decay=1e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

**策略**: 新分类器参数以 10 倍学习率训练，backbone 微调使用较低学习率。

### 7.6 损失函数

```python
criterion = nn.CrossEntropyLoss()
```

标准多分类交叉熵。每个 segment 的标签为对应说话人的 ID，模型输出 `(B, num_speakers)` 分类 logits。

### 7.7 训练循环

```
for epoch in range(epochs):
    # Train
    model.train()
    for batch in train_loader:
        outputs = model(feats)          # 前向
        loss = criterion(outputs, labels)   # 交叉熵
        loss.backward()                 # 反向传播
        clip_grad_norm_(5.0)            # 梯度裁剪 (防梯度爆炸)
        optimizer.step()                # 参数更新
    scheduler.step()                    # 余弦退火

    # Validation
    model.eval()
    with torch.no_grad():
        计算验证集 accuracy
    保存最佳模型 (最高 val acc)
```

- **梯度裁剪**: `max_norm=5.0`，防止低资源微调时的梯度爆炸
- **余弦退火调度**: 学习率从初始值余弦衰减到 0
- **早停**: 手动追踪 `best_val_acc`，仅保存超过历史最佳的模型

### 7.8 模型导出

```python
# 完整模型（含分类器）
torch.save(model.state_dict(), f"{model_name}_final.pt")

# Backbone-only（去除分类器，用于部署推理）
model.cpu()
del model.classifier  # 或 projection
torch.save(model.state_dict(), f"{model_name}_backbone.pt")
```

---

## 8. ONNX 推理模型

生产环境使用 ONNX 模型进行推理，部署在 `app/api/models/`:

| 模型 | ONNX 文件名 | 嵌入维度 |
|---|---|---|
| CAM++ | `campplus.onnx` | 192 |
| ResNet34 | `voxceleb_resnet34_LM.onnx` | 256 |
| ECAPA | `ecapa-speaker-v1.onnx` | 192 |

**推理流程** (见 `app/api/services/verifier.py`):
1. 加载音频 → 重采样 16kHz → fbank 提取
2. ONNX Runtime 推理 → embedding
3. 缓存 embedding（按 audio_id 或 URL MD5 缓存）
4. Cosine/欧氏/点积 相似度计算
5. 阈值决策（CAM++=0.49, ResNet34=0.59, ECAPA=0.68）

---

## 9. 模型版本管理与发布流程

**文件**: `app/train/model_manager.py`, `app/train/incremental_train.py`, `app/train/evaluator.py`

### 9.1 发布条件 (`evaluator.should_publish`)

```python
should_publish(new_eer, prev_eer, improvement_threshold=0.001)
```

- 首版模型 → 直接发布
- EER 降低 > 0.001 → 发布
- 否则跳过

### 9.2 版本编号

```
get_next_version(conn):
  active = 'v1.0' → 'v1.1'
  active = 'v1.1' → 'v1.2'
```

### 9.3 发布流程 (model_manager.publish_model)

1. 复制 ONNX 到 `api/models/{version}.onnx`
2. 替换 `api/models/campplus.onnx`（热加载）
3. 注册数据库记录（含 MD5、EER、训练统计）

### 9.4 评估指标 (evaluator.py)

```python
compute_eer(scores, labels)       # EER (%) + 对应阈值
compute_min_dcf(scores, labels,   # minDCF@P_target
                C_miss=1.0, C_fa=1.0, P_target=0.01)
```

EER 计算方式：分数降序排列 → 累积 FAR/FRR → 找到 FAR≈FRR 的点。

---

## 10. 模型对比 (evaluate.py)

**文件**: `app/train/evaluate.py`

### 10.1 用途

对比 **fine-tune 前（预训练）** 与 **fine-tune 后（微调）** 的 embedding 质量。

### 10.2 核心指标

**Separation = Within-class mean - Between-class mean**

| 指标 | 含义 |
|---|---|
| `within['mean']` | 同人段间 cosine 相似度均值 |
| `between['mean']` | 异人段间 cosine 相似度均值 |
| `separation` | 两者的差值（越大越好） |
| `within Δ` | fine-tune 后同人相似度变化（正=改善） |
| `between Δ` | fine-tune 后异人相似度变化（负=更好区分） |

### 10.3 使用方法

```bash
PYTHONPATH=. python app/train/evaluate.py --model all --epoch 20
```

输出示例：
```
Model: resnet
  [Pretrained]  Within=0.6250  Between=0.2316  Separation=0.3934
  [Fine-tuned] Within=0.7485  Between=0.2212  Separation=0.5273
  [Delta]      Within Δ=+0.1235  Between Δ=-0.0104  Sep Δ=+0.1339
```

### 10.4 说话人级别分析

评估同时计算每个说话人的 within-class 相似度变化，并汇总 improve/degrade 计数。

---

## 附录: 文件清单

| 文件 | 内容 |
|---|---|
| `app/pytorch_models/campp_model.py` | CAM++ 模型定义 + 权重加载 |
| `app/pytorch_models/resnet34_model.py` | ResNet34 模型定义 + 权重加载 |
| `app/pytorch_models/ecapa_model.py` | ECAPA-TDNN 模型定义 + 权重加载 |
| `app/pytorch_models/components.py` | 公共网络层组件 |
| `app/pytorch_models/verify_weights.py` | 验证三个模型权重加载的测试脚本 |
| `app/train/fine_tune.py` | 训练主流程（数据加载→训练→导出） |
| `app/train/evaluate.py` | 前后对比评估脚本 |
| `app/train/evaluator.py` | EER/minDCF 计算工具 |
| `app/train/model_manager.py` | 模型版本管理与发布 |
| `app/train/config.py` | 训练配置加载 |
| `app/api/services/verifier.py` | 生产环境推理服务 |
