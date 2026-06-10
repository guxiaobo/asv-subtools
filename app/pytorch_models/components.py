"""Standalone model layers adapted from egrecho.nn.components + 3D-Speaker."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def Nonlinearity(nl: str) -> nn.Module:
    if nl == "relu":
        return nn.ReLU(inplace=True)
    elif nl == "mish":
        return Mish()
    elif nl == "swish":
        return Swish()
    elif nl == "silu":
        return nn.SiLU(inplace=True)
    elif nl == "linear" or nl == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown nonlinearity: {nl}")


def get_bn_relu(channels, eps=1e-5):
    return nn.Sequential(OrderedDict([
        ("batchnorm", nn.BatchNorm1d(channels, eps=eps)),
        ("relu", Nonlinearity("relu")),
    ]))


def statistics_pooling(x: torch.Tensor, dim: int = -1, keepdim: bool = False, unbiased: bool = True):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class TDNNBlock(nn.Module):
    """Matched to 3D-Speaker naming (OrderedDict nonlinear for batchnorm key)."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 pad=True, dilation=1, bias=True, pre_norm=False):
        super().__init__()
        if pad:
            assert kernel_size % 2 == 1
            padding = (kernel_size - 1) // 2 * dilation
        else:
            padding = 0
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, bias=bias)
        if pre_norm:
            self.nonlinear = nn.Sequential(OrderedDict([
                ("batchnorm", nn.BatchNorm1d(out_channels)),
                ("relu", Nonlinearity("relu")),
            ]))
        else:
            self.nonlinear = nn.Sequential(OrderedDict([
                ("relu", Nonlinearity("relu")),
                ("batchnorm", nn.BatchNorm1d(out_channels)),
            ]))

    def forward(self, x):
        unsqueezed = False
        if x.ndim == 2:
            x = x.unsqueeze(dim=2)
            unsqueezed = True
        x = self.linear(x)
        x = self.nonlinear(x)
        return x.squeeze(2) if unsqueezed else x


class DenseLayer(TDNNBlock):
    """1x1 conv + BN (no bias, no affine) for final embedding."""
    def __init__(self, in_channels, out_channels, bias=False, affine=False):
        super().__init__(in_channels, out_channels, kernel_size=1,
                         bias=bias, pre_norm=True)
        # Override batchnorm affine=False to match checkpoint
        self.nonlinear[0] = nn.BatchNorm1d(out_channels, affine=affine)


class SERes2Block(nn.Module):
    """SE-Res2Block from Wespeaker ECAPA."""
    def __init__(self, channels, kernel_size=3, dilation=1, scale=8):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.se_res2block = nn.ModuleList()
        for s in range(scale):
            self.se_res2block.append(nn.Sequential(OrderedDict([
                ("conv", nn.Conv1d(channels // scale, channels // scale, kernel_size,
                                   padding=dilation, dilation=dilation, bias=False)),
                ("bns", nn.ModuleList([
                    nn.BatchNorm1d(channels // scale)
                    for _ in range(2)]
                )),
            ])))
        self.se_fc1 = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.se_fc2 = nn.Conv1d(channels // 8, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn(self.conv(x)))
        # Split and process scales
        sc = torch.chunk(x, len(self.se_res2block), dim=1)
        outs = []
        for i, s in enumerate(sc):
            out = self.se_res2block[i][0](s)
            out = self.se_res2block[i][1][0](out)
            out = F.relu(out)
            out = self.se_res2block[i][1][1](out)
            out = F.relu(out)
            outs.append(out)
        x = torch.cat(outs, dim=1)
        # SE
        se = x.mean(dim=-1, keepdim=True)
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        x = x * se
        return x + residual
