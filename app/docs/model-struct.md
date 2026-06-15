# 三模型声纹提取网络 — 架构与训练算法详解

> 基于 `app/train/fine_tune.py`（增量训练脚本）中的精确模型定义，
> 每个 checkpoint 的架构已通过 keys/shapes 验证，精确匹配原始预训练权重。

---

## 1. CAM++

**文件:** `app/train/fine_tune.py` 第 293 行 `class CAMPlus`
**预训练权重:** `app/pytorch_weights/campplus_cn_common.pt`
**嵌入维度:** 192
**来源:** Alibaba 3D-Speaker / egrecho CamPP

### 1.1 架构总览

```
FCM(2D前端, feat_dim=80)
    → 输出 channels = 32 × (80÷8) = 320
    → TDNN(320→128, k=5, stride=2)       # tdnn
    → 3×CAMDenseTDNNBlock + TransitLayer  # block1~3 + transit1~3
        block1: 12层, k=3, d=1 → transit(688→344)
        block2: 24层, k=3, d=2 → transit(1112→556)
        block3: 16层, k=3, d=2 → transit(1068→534)
    → BN + ReLU (out_nonlinear)
    → StatsPool (mean+std) → 1068
    → DenseLayer(1068→192)               # dense
    → Linear(192, num_speakers)           # projection（训练时才存在）
```

### 1.2 组件详解

**FCM（Front-end Convolution Module，第 99 行）**
- 2D 前端：`Conv2d(1→32, k=3) + BN + ReLU`
- 两层 BasicResBlock2D（stride=2 频域降采样）
- 再一层 `Conv2d(32→32, k=3, stride=(2,1))`
- 最终 reshape 合并通道×频域：`shape[1]*shape[2]`
- 等效频域降采样率 = 8

**BasicResBlock2D（第 72 行）**
- 标准 2D 残差块，stride=(s,1) 只在频域方向降采样
- conv1 → BN → ReLU → conv2 → BN → shortcut → ReLU
- expansion=1

**CAMLayer（Context-Aware Masking，第 135 行）**
- 本地路径：`Conv1d + 分段池化 + 全局上下文`
- 全局 sigmoid mask 调制本地卷积输出
- `seg_pooling`：avg/max 分段池化（seg_len=100 帧），插值回原长度
- 瓶颈缩减比 reduction=2

**CAMDenseTDNNBlock（第 190 行）**
- Dense 连接堆叠：每层输入 = 前几层输出的 concat
- 每层：BN+ReLU → Conv1d 1×1 → BN+ReLU → CAMLayer
- growth_rate=32, bn_size=4

**TransitLayer（第 213 行）**
- BN+ReLU → Conv1d 1×1（通道数减半）
- 控制维度增长，放在每个 Dense Block 之后

**StatsPool（第 226 行）**
- 时间维度的 mean + std 拼接

**DenseLayer（第 272 行）**
- 纯 `Conv1d 1×1` 输出嵌入向量
- **注意：** 去掉了原始实现中的 BN（因 batch_size=1 时 1×1 conv 输出 BN 零方差）

### 1.3 前向传播

```python
def forward(self, x, return_embedding=False):
    # x: (B, T, F)  # 模型要求的原始输入格式
    x = x.permute(0, 2, 1)      # (B, F, T)
    x = self.head(x)               # FCM → (B, 320, T)
    x = self.xvector(x)            # Dense blocks → StatsPool → Dense → (B, 192)
    if return_embedding or self.projection is None:
        return x
    return self.projection(x)
```

---

## 2. ECAPA-TDNN

**文件:** `app/train/fine_tune.py` 第 447 行 `class ECAPA_TDNNSpeaker`
**预训练权重:** `app/pytorch_weights/avg_model.pt`
**嵌入维度:** 192
**来源:** 标准 ECAPA-TDNN 实现

### 2.1 架构总览

```
input: (B, F, T)
    → layer1: TDNN(80→512, k=5) + BN + ReLU
    → layer2: SE-Res2Block(512, d=2)
    → layer3: SE-Res2Block(512, d=3)
    → layer4: SE-Res2Block(512, d=4)
    → concat(layer2, layer3, layer4) → (B, 1536, T)
    → conv: Conv1d(1536→1536, k=1)     # MFA
    → pool: AttentiveStatsPool(1536, hidden=128, time_attention=True)
    → bn: BN(3072)
    → linear: Linear(3072→192)
    → projection: Linear(192, num_speakers)
```

### 2.2 组件详解

**Res2NetConvBlock（第 372 行）**
- 将输入按 scale=8 分块（chunk into 8 groups）
- 每个子块：Conv1d(out_ch→out_ch, k=3) + BN + ReLU
- 残差连接：`sp = sp + xs[i+1]`
- 最后 concat 所有子块输出

**SEModule（第 399 行）**
- Squeeze: 全局平均池化 → Linear(ch→128)
- Excitation: ReLU → Linear(128→ch) → Sigmoid
- 输出 = 输入 × attention 权重

**SE_Res2Block（第 413 行）**
- ModuleList 中有序包含 4 个组件：
  - `[0]`: Conv1d(512→512, k=1) + BN（1×1 卷积降维）
  - `[1]`: Res2NetConvBlock（分组卷积）
  - `[2]`: Conv1d(512→512, k=1) + BN（1×1 卷积恢复维度）
  - `[3]`: SEModule(512→128→512)
- 残差连接：`x + residual`

**AttentiveStatsPoolECAPA（第 507 行）**
- `time_attention=True` 模式：attention 输入 = `concat(x, global_mean, global_std)`，维度 3×in_dim
- `linear1: Conv1d(3×in_dim→128, k=1) → Tanh → linear2: Conv1d(128→in_dim, k=1) → Softmax`
- 加权 mean + 加权 std（`Σαx² - (Σαx)²` 的平方根）

### 2.3 关键细节

- 所有 BN 使用 `momentum=0.5`（与原始 checkpoint 一致）
- MFA 层（multi-layer feature aggregation）直接 concat 三层 SE-Res2Block 输出
- 残差路径：`x2 = layer2(x)`, `x3 = layer3(x + x1)`, `x4 = layer4(x + x1 + x2)`

---

## 3. ResNet34 (2D)

**文件:** `app/train/fine_tune.py` 第 567 行 `class ResNet34_2D`
**预训练权重:** `app/pytorch_weights/avg_model`（无后缀）
**嵌入维度:** 256
**来源:** 标准 2D ResNet34 适配声纹

### 3.1 架构总览

```
input: (B, 1, F, T)
    → conv1: Conv2d(1→32, k=3×3, stride=1, padding=1) + BN + ReLU
    → layer1: 3 × BasicBlock2D(32→32, stride=1)
    → layer2: 4 × BasicBlock2D(32→64, stride=2)
    → layer3: 6 × BasicBlock2D(64→128, stride=2)
    → layer4: 3 × BasicBlock2D(128→256, stride=2)
    → reshape: (B, 256×H, T)     # H = ceil(80/8) = 10
    → StatsPool: (B, 5120)       # mean + std = 2×2560
    → seg_1: Linear(5120→256)    # 嵌入层
    → projection: Linear(256, num_speakers)
```

### 3.2 降采样与维度计算

- 三层 stride=2 的 layer，频域总降采样率 = 2³ = 8
- feat_dim=80 → H = ceil(80/8) = 10
- StatsPool 输入: (B, 256×10, T) = (B, 2560, T)
- StatsPool 输出: (B, 5120) = (B, 2×2560)
- 嵌入层: `Linear(5120, 256)`，固定输出 256 维

### 3.3 BasicBlock2D（第 540 行）

- 标准 2D 残差块
- conv1(k=3, stride) → BN(m=0.5) → ReLU → conv2(k=3, stride=1) → BN(m=0.5) → +shortcut → ReLU
- stride≠1 时 1×1 卷积 shortcut

---

## 4. FBank 特征提取

**文件:** `app/train/fine_tune.py` 第 642 行 `class FBankExtractor`

纯 PyTorch 实现，无 torchaudio 依赖：

| 参数 | 值 | 说明 |
|------|-----|------|
| n_mels | 80 | 梅尔滤波器数量 |
| sr | 16000 | 采样率 |
| n_fft | 512 | STFT 窗口大小 |
| hop_length | 160 | 帧移（10ms） |
| win_length | 400 | 窗口长度（25ms） |
| f_min / f_max | 0 / 8000 | 频率范围 |

预处理流程：Hann 窗 → STFT → |mag|² → mel_filterbank → log → (T, 80)

---

## 5. 训练算法

### 5.1 数据构建 (`build_training_data`, 第 745 行)

1. 从 SQLite `recordings` 表查询 `status='preprocessed'` 的录音
2. 从 `audio_segments` 表查询所有段
3. 对每通录音的 VAD 段，用 `SpeakerDiarizer`（Otsu 双峰法）标注坐席/客户
4. 只保留标注为 `customer` 的段作为训练数据
5. 按客户电话号码（customer_phone）作为说话人 ID
6. 返回 `[(wav_path, speaker_id), ...]` 列表 + `id_to_speaker` 映射

### 5.2 数据集 (`SpeakerDataset`, 第 710 行)

- 滑动窗口加载：每条段随机裁剪/pad 到 400 帧（4 秒 @ 10ms）
- 段 < 400 帧 → pad；段 > 400 帧 → 随机切取 400 帧

### 5.3 优化器 (第 967 行)

**两阶段 Adam：**
```python
optimizer = torch.optim.Adam([
    {'params': backbone_params, 'lr': 1e-4},   # backbone 参数
    {'params': classifier_params, 'lr': 1e-3},  # 分类头（10× lr）
], weight_decay=1e-4)
```

- backbone 参数：所有不包含 `classifier` 或 `projection` 的参数
- classifier 参数：随 num_speakers 变化的分类头参数
- 分类头学习率是 backbone 的 10 倍，加速新类别适配

### 5.4 学习率调度 (第 981 行)

- `CosineAnnealingLR(optimizer, T_max=epochs)`
- 从 1e-4（backbone）/ 1e-3（classifier）余弦衰减到 0

### 5.5 损失函数

- `CrossEntropyLoss`（标准分类损失）

### 5.6 梯度裁剪 (第 1000 行)

- `clip_grad_norm_(model.parameters(), 5.0)`
- 防止 RNN/Conv1d 梯度爆炸

### 5.7 训练循环 (第 988 行)

```
for epoch in range(epochs):
    model.train()
    for feats, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(feats)
        loss = criterion(outputs, labels)
        loss.backward()
        clip_grad_norm_(5.0)
        optimizer.step()
    scheduler.step()
    # Validation + save best model
```

### 5.8 验证与模型保存 (第 1010~1053 行)

- 每 epoch 验证，保存 val_acc 最高的模型
- **三份输出权重:**
  1. `{model_name}_best.pt` — 验证集最佳 epoch
  2. `{model_name}_final.pt` — 最后一 epoch（完整含 projection）
  3. `{model_name}_backbone.pt` — 去掉 projection 后的 backbone（推理用）

### 5.9 模型版本入库 (第 1074 行)

训练结束后自动调用 `_register_trained_model()`：
- 写入 `model_versions` 表
- 记录：版本标签、嵌入维度、验证准确率、路径、MD5、训练段数和说话人数

---

## 6. 交互维度对比

| 特性 | CAM++ | ECAPA-TDNN | ResNet34 |
|------|-------|------------|----------|
| 嵌入维度 | 192 | 192 | 256 |
| 前端处理 | FCM (2D Conv) | Conv1d k=5 | Conv2d k=3×3 |
| 主体 | DenseTDNN + CAM | SE-Res2Block ×3 | ResNet34 (2D) |
| 池化 | StatsPool (mean+std) | AttentiveStatsPool | StatsPool (mean+std) |
| 注意力机制 | CAM (Context-Aware Mask) | SE (Squeeze-Excitation) | 无 |
| 特征聚合 | Dense concat | MFA (所有 layer concat) | 通道-频域合并 reshape |
| 参数量（约） | 7.2M | 6.5M | 4.5M |
| 电话录音表现 | 中等（需低聚类阈值） | 较强（高 sim threshold） | 较强（高 sim threshold） |
| 坐席判定默认阈值 | 0.49 | 0.68 | 0.59 |
| 客户聚类默认阈值 | 0.35 | 0.55 | 0.55 |

---

## 7. 预训练权重加载

所有模型通过 `_load_backbone()` 加载（第 352 行）：
1. 如果 key 以 `module.` 开头 → 去掉前缀（兼容 DDP 格式）
2. 过滤掉 `projection.*` / `classifier.*` key（维度随 num_speakers 变化）
3. `model.load_state_dict(filtered, strict=False)`
4. 报告 missing/unexpected keys
