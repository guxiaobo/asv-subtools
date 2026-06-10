"""ECAPA-TDNN model matching Wespeaker pretrained weights.
Based on Wespeaker (wav2vec-speaker) implementation.
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from components import Nonlinearity, get_bn_relu


class SE_Res2Block(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, base_width=64, scale=8, se_ratio=8):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.scale = scale
        self.width = base_width
        self.se_res2block = nn.ModuleList()
        for i in range(scale):
            self.se_res2block.append(nn.ModuleList([
                nn.Conv1d(channels // scale, channels // scale, kernel_size,
                          padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm1d(channels // scale),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(channels // scale),
                nn.ReLU(inplace=True),
            ]))
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(channels)
        self.se_fc1 = nn.Conv1d(channels, channels // se_ratio, kernel_size=1)
        self.se_fc2 = nn.Conv1d(channels // se_ratio, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        # Scaled split
        chunks = torch.chunk(x, self.scale, dim=1)
        outs = []
        for i, c in enumerate(chunks):
            out = self.se_res2block[i][0](c)
            out = self.se_res2block[i][1](out)
            out = self.se_res2block[i][2](out)
            out = self.se_res2block[i][3](out)
            out = self.se_res2block[i][4](out)
            outs.append(out)
        x = torch.cat(outs, dim=1)
        x = F.relu(self.bn3(self.conv3(x)))
        # SE module
        se = x.mean(dim=-1, keepdim=True)
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        x = x * se
        return x + residual


class EcapaTdnn(nn.Module):
    """Wespeaker ECAPA-TDNN model.
    Matches checkpoint naming: layer1.*, layer2.se_res2block.*, bn.*, conv.*
    """
    def __init__(self, inputs_dim=80, channels=512, embd_dim=192, mfa_dim=1536):
        super().__init__()
        self.layer1 = nn.Sequential(OrderedDict([
            ("conv", nn.Conv1d(inputs_dim, channels, kernel_size=5, padding=2)),
            ("bn", nn.BatchNorm1d(channels)),
        ]))
        # SE-Res2Block layers
        self.layer2 = nn.ModuleList()
        for dilation in [1, 2, 3]:
            self.layer2.append(SE_Res2Block(channels, dilation=dilation))
        # MFA + output 
        self.conv = nn.Conv1d(channels * 3, mfa_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(mfa_dim)
        # Attentive Stats Pooling defined outside (no params in state_dict naming)

    def forward(self, x):
        # Input: (B, T, F) -> (B, F, T)
        x = x.permute(0, 2, 1).contiguous()
        x = F.relu(self.layer1(x))
        # Collect multi-scale outputs
        outputs = [self.layer2[i](x) for i in range(3)]
        x = torch.cat(outputs, dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        # Attentive stats pooling + embedding layer
        # (to be handled by EcapaModel wrapper)
        return x


class EcapaModel(nn.Module):
    """Full ECAPA model with attentive stats pooling + embedding layer.

    Args:
        inputs_dim: Input feature dim (default: 80)
        channels: Model channels (default: 512)
        embd_dim: Output embedding dim (default: 192)
        mfa_dim: MFA layer dim (default: 1536)
    """
    def __init__(self, inputs_dim=80, channels=512, embd_dim=192, mfa_dim=1536):
        super().__init__()
        self.embd_dim = embd_dim
        self.ecapa = EcapaTdnn(inputs_dim=inputs_dim, channels=channels,
                                embd_dim=embd_dim, mfa_dim=mfa_dim)
        # Attention stats pooling
        self.attention = nn.Sequential(OrderedDict([
            ("linear1", nn.Conv1d(mfa_dim, 128, kernel_size=1)),
            ("tanh", nn.Tanh()),
            ("linear2", nn.Conv1d(128, mfa_dim, kernel_size=1)),
            ("softmax", nn.Softmax(dim=-1)),
        ]))
        self.bn = nn.BatchNorm1d(mfa_dim * 2)
        self.fc = nn.Linear(mfa_dim * 2, embd_dim)
        self.bn_out = nn.BatchNorm1d(embd_dim)

    def forward(self, input_features):
        x = self.ecapa(input_features)
        # Attentive statistics pooling
        att = self.attention(x)
        mean = (x * att).sum(dim=-1)
        var = ((x ** 2) * att).sum(dim=-1) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-10))
        stats = torch.cat([mean, std], dim=1)
        stats = self.bn(stats)
        x = self.fc(stats)
        x = self.bn_out(x)
        return x


def create_ecapa_model(embd_dim=192, pretrained_path=None):
    """Create ECAPA model with optional pretrained weights.

    Loads Wespeaker avg_model.pt checkpoint.
    The checkpoint has EcapaTdnn + attention + bn_fc layers.
    """
    model = EcapaModel(embd_dim=embd_dim)
    load_status = {"loaded": False, "missing": [], "unexpected": []}

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        filtered = {k: v for k, v in state_dict.items()
                    if not k.endswith("num_batches_tracked")}
        # Map Wespeaker keys to our model
        # Wespeaker has: layer1.conv, layer2.se_res2block.N.0-4, conv, bn, attention, bn2, fc, bn3
        # Our model: ecapa.layer1.*, ecapa.layer2.N.*, ecapa.conv, ecapa.bn, attention.*, bn, fc, bn_out
        mapping = {}
        for k, v in filtered.items():
            if k.startswith("layer"):
                mapping[f"ecapa.{k}"] = v
            elif k.startswith("conv"):
                mapping[f"ecapa.{k}"] = v
            elif k.startswith("bn"):
                mapping["bn_out.weight"] = filtered.get("bn3.weight")
                mapping["bn_out.bias"] = filtered.get("bn3.bias")
                mapping["bn_out.running_mean"] = filtered.get("bn3.running_mean")
                mapping["bn_out.running_var"] = filtered.get("bn3.running_var")
                if k == "bn":
                    mapping["ecapa.bn.weight"] = v
                    mapping["ecapa.bn.running_mean"] = filtered.get(k.replace("weight", "running_mean"))
                    mapping["ecapa.bn.running_var"] = filtered.get(k.replace("weight", "running_var"))
                    mapping["ecapa.bn.bias"] = filtered.get(k.replace("weight", "bias"))
                    continue
                break
            elif k.startswith("attention"):
                mapping[k] = v
            elif k.startswith("bn2"):
                mapping["bn.weight"] = v
                mapping["bn.bias"] = filtered.get("bn2.bias")
                mapping["bn.running_mean"] = filtered.get("bn2.running_mean")
                mapping["bn.running_var"] = filtered.get("bn2.running_var")
            elif k.startswith("fc"):
                mapping[k] = v
            elif k.startswith("bn3"):
                mapping["bn_out.weight"] = v
                mapping["bn_out.bias"] = filtered.get("bn3.bias")
                mapping["bn_out.running_mean"] = filtered.get("bn3.running_mean")
                mapping["bn_out.running_var"] = filtered.get("bn3.running_var")

        try:
            missing, unexpected = model.load_state_dict(mapping, strict=False)
            load_status["missing"] = [k for k in missing if "num_batches_tracked" not in k]
            load_status["unexpected"] = unexpected
            load_status["loaded"] = len(load_status["missing"]) == 0
        except Exception as e:
            load_status["error"] = str(e)

    return model, load_status
