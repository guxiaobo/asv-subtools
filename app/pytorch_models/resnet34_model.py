"""ResNet34 speaker model matching Wespeaker pretrained weights.
Based on the ResNet34-LM from WeSpeaker.
Channels: 32 base, 3-layer groups, final 256-dim embedding.
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from components import StatsPool


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class ResNet34(nn.Module):
    """ResNet34 speaker embedding model.

    Base channels=32, layer_groups=[3,4,6,3], no avgpool stride.
    Output: 256-dim embedding (or as specified).
    """
    def __init__(self, inputs_dim=80, embd_dim=256, base_channels=32):
        super().__init__()
        self.embd_dim = embd_dim
        self.in_planes = base_channels

        # Front-end: conv1d that simulates the original 2D conv
        # Wespeaker uses 2D conv, but our ONNX uses 1D time projection
        # The pretrained avg_model has 2D conv layers
        # We match the checkpoint naming exactly
        self.conv1 = nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.layer1 = self._make_layer(BasicBlock, base_channels, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, base_channels * 2, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_channels * 4, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, base_channels * 8, 3, stride=2)

        last_channels = base_channels * 8 * BasicBlock.expansion  # 256
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_channels, embd_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input: (B, T, F) -> (B, 1, F, T)
        x = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, F, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_resnet34_model(embd_dim=256, pretrained_path=None):
    """Create ResNet34 model with optional pretrained weights.

    Load Wespeaker avg_model checkpoint.
    """
    model = ResNet34(embd_dim=embd_dim)
    load_status = {"loaded": False, "missing": [], "unexpected": []}

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        filtered = {k: v for k, v in state_dict.items()
                    if not k.endswith("num_batches_tracked")}
        try:
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            load_status["missing"] = [k for k in missing if "num_batches_tracked" not in k]
            load_status["unexpected"] = unexpected
            load_status["loaded"] = len(load_status["missing"]) == 0
        except Exception as e:
            load_status["error"] = str(e)

    return model, load_status
