'''
This file is modified fron https://github.com/luuuyi/CBAM.PyTorch by Chen Wei
'''

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import build_norm_layer


class ChannelAttention(BaseModule):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(BaseModule):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(BaseModule):
    def __init__(self, planes, init_cfg=None):
        super(CBAM, self).__init__(init_cfg)
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class CBAMWithPosEmbed(BaseModule):
    def __init__(self, planes, norm_cfg, pos_embed_planes=16, init_cfg=None):
        super(CBAMWithPosEmbed, self).__init__(init_cfg)
        self.pos_embed_planes = pos_embed_planes
        self.ca = ChannelAttention(pos_embed_planes)
        self.sa = SpatialAttention()
        self.pos_embed_mlp = nn.Sequential(
            nn.Linear(2, 8, bias=True),
            nn.ReLU(True),
            nn.Linear(8, pos_embed_planes, bias=True),
            nn.ReLU(True),
        )
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(planes, pos_embed_planes, 3, 1, 1, bias=False),
            build_norm_layer(norm_cfg, pos_embed_planes)[1],
            nn.ReLU(inplace=True),
        )
        self.dim_increase = nn.Sequential(
            nn.Conv2d(pos_embed_planes, planes, 1, 1, 0, bias=False),
            build_norm_layer(norm_cfg, planes)[1],
            nn.ReLU(inplace=True),
        )

    def pos_embed(self, feature, eps=1e-7):
        bn, ch, h, w = feature.shape
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w))
        pos = torch.stack((xx, yy), axis=-1).float().to(feature.device)
        pos = pos.div(pos.new_tensor([w, h])) - 0.5
        assert pos.shape == (h, w, 2)
        f = self.pos_embed_mlp(pos.view(-1, 2))
        f = f.view(h, w, self.pos_embed_planes)
        return f.permute(2, 0, 1).contiguous()

    def forward(self, x) -> torch.Tensor:
        x_r = self.dim_reduce(x)
        x_r = x_r + self.pos_embed(x)
        x = x * self.dim_increase(self.ca(x_r))
        x = x * self.sa(x_r)
        return x


