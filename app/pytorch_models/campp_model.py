"""CAM++ model (standalone PyTorch).
Architecture matches 3D-Speaker jingyaogong/campplus pretrained weights.
Adapted from egrecho/models/campplus/campplus.py
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from components import TDNNBlock, DenseLayer, StatsPool, get_bn_relu, Nonlinearity, Mish


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(nn.Module):
    """Front-end CNN module for CAM++."""
    def __init__(self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, F, T) -> (B, 1, F, T)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))
        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMLayer(nn.Module):
    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding, dilation, bias, reduction=2):
        super().__init__()
        self.linear_local = nn.Conv1d(bn_channels, out_channels, kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(shape[0], shape[1], shape[2], seg_len).reshape(shape[0], shape[1], -1)
        seg = seg[..., :x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bn_channels, kernel_size,
                 stride=1, dilation=1, bias=False):
        super().__init__()
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2 * dilation
        self.nonlinear1 = get_bn_relu(in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_bn_relu(bn_channels)
        self.cam_layer = CAMLayer(bn_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.linear1(self.nonlinear1(x))
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self, num_layers, in_channels, out_channels, bn_channels,
                 kernel_size, stride=1, dilation=1, bias=False):
        super().__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
            )
            self.add_module(f"tdnnd{i + 1}", layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.nonlinear = get_bn_relu(in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class CamPP(nn.Module):
    """CAM++ model - standalone PyTorch.

    Args:
        inputs_dim: Input feature dimension (default: 80 for fbank)
        embd_dim: Output embedding dimension (default: 192)
        init_channels: Initial channels (default: 128)
        growth_rate: Per-layer growth rate (default: 32)
        bn_size: Bottleneck multiplier (default: 4)
    """
    def __init__(self, inputs_dim=80, embd_dim=192, init_channels=128,
                 growth_rate=32, bn_size=4):
        super().__init__()
        self.head = FCM(feat_dim=inputs_dim)
        channels = self.head.out_channels

        self.xvector = nn.Sequential(OrderedDict([
            ("tdnn", TDNNBlock(channels, init_channels, 5, stride=2, dilation=1, bias=False, pre_norm=True)),
        ]))
        channels = init_channels

        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            self.xvector.add_module(f"block{i + 1}", block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(f"transit{i + 1}", TransitLayer(channels, channels // 2, bias=False))
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_bn_relu(channels))
        self.xvector.add_module("stats", StatsPool())
        self.xvector.add_module("dense", DenseLayer(channels * 2, embd_dim))

    def forward(self, input_features):
        # input: (B, T, F) -> (B, F, T)
        x = input_features.permute(0, 2, 1)
        x = self.head(x)
        x = self.xvector(x)
        return x


def create_campp_model(embd_dim=192, pretrained_path=None):
    """Create CAM++ model with optional pretrained weights.

    Returns:
        model: CamPP module (returns embedding of size embd_dim)
        load_status: dict with status information
    """
    model = CamPP(embd_dim=embd_dim)
    load_status = {"loaded": False, "missing": [], "unexpected": []}

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        # Filter out num_batches_tracked (optional in newer PyTorch)
        filtered = {k: v for k, v in state_dict.items()
                    if not k.endswith("num_batches_tracked")}
        # Try strict load first
        try:
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            load_status["missing"] = [k for k in missing if "num_batches_tracked" not in k]
            load_status["unexpected"] = unexpected
            load_status["loaded"] = len(load_status["missing"]) == 0
        except Exception as e:
            load_status["error"] = str(e)

    return model, load_status
