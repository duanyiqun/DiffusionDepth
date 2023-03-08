# Copyright (c) Phigent Robotics. All rights reserved.
# for debug
# import sys
# sys.path.append('/mnt/cfs/algorithm/yiqun.duan/depth/diffuserdc/src')

import torch
from torch import nn
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock
import torch.utils.checkpoint as checkpoint
from mmdet.models import BACKBONES
from mmcv.runner import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from model.ops.cbam import CBAMWithPosEmbed


class BasicBlockWithCBAM(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlockWithCBAM, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.cbam = CBAMWithPosEmbed(planes, norm_cfg, pos_embed_planes=min(planes, 16), init_cfg=init_cfg)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.cbam(out)
            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = checkpoint.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNetForMMBEV(nn.Module):
    def __init__(self, numC_input, num_layer=[2, 2, 2], num_channels=None, stride=[2, 2, 2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic', ):
        super(ResNetForMMBEV, self).__init__()
        # build backbone
        # assert len(num_layer)>=3
        assert len(num_layer) == len(stride)
        num_channels = [numC_input * 2 ** (i + 1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [Bottleneck(curr_numC, num_channels[i] // 4, stride=stride[i],
                                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([Bottleneck(curr_numC, curr_numC // 4,
                                         norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [BasicBlock(curr_numC, num_channels[i], stride=stride[i],
                                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'BasicBlockWithCBAM':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [BasicBlockWithCBAM(curr_numC, num_channels[i], stride=stride[i],
                                            downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                            norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend(
                    [BasicBlockWithCBAM(curr_numC, curr_numC, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


class IdentityMMBEV(nn.Module):
    def __init__(self, numC_input, num_layer=[2, 2, 2], num_channels=None, stride=[2, 2, 2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic', ):
        super(IdentityMMBEV, self).__init__()
        # build backbone
        # assert len(num_layer)>=3

    def forward(self, x):
        x_tmp = x
        return x_tmp


def mmbev_res18():
    net = ResNetForMMBEV(3, num_layer=[2, 2, 2, 2], num_channels=[64, 128, 256, 512], stride=[2, 2, 2, 2],
                         backbone_output_ids=None, norm_cfg=dict(type='BN'),
                         with_cp=False, block_type='Basic', )
    return net


def mmbev_res50():
    net = ResNetForMMBEV(3, num_layer=[3, 4, 6, 3], num_channels=[64, 128, 256, 512], stride=[2, 2, 2, 2],
                         backbone_output_ids=None, norm_cfg=dict(type='BN'),
                         with_cp=False, block_type='Basic', )
    return net


def mmbev_res101():
    net = ResNetForMMBEV(3, num_layer= [3, 4, 23, 3], num_channels=[64, 128, 256, 512], stride=[2, 2, 2, 2],
                         backbone_output_ids=None, norm_cfg=dict(type='BN'),
                         with_cp=False, block_type='Basic', )
    return net


def indentity():
    net = IdentityMMBEV(3, num_layer= [3, 4, 23, 3], num_channels=[64, 128, 256, 512], stride=[2, 2, 2, 2],
                         backbone_output_ids=None, norm_cfg=dict(type='BN'),
                         with_cp=False, block_type='Basic', )
    return net



if __name__ == '__main__':
    test_sample = torch.randn(1, 3, 228, 304)
    model = ResNetForMMBEV(3, num_layer=[2, 2, 2, 2], num_channels=[64, 128, 256, 512], stride=[2, 2, 2, 2],
                           backbone_output_ids=None, norm_cfg=dict(type='BN'),
                           with_cp=False, block_type='Basic', )
    fp = model(test_sample)
    for f in fp:
        print(f.shape)

    """ output sample 
    test_sample = torch.randn(1, 3, 228, 304)
    
    output list 
    torch.Size([1, 64, 114, 152])
    torch.Size([1, 128, 57, 76])
    torch.Size([1, 256, 29, 38])
    torch.Size([1, 512, 15, 19])
    """

    """ output sample 
        test_sample = torch.randn(1, 3, 228, 304)

        output list 
        torch.Size([1, 64, 114, 152])
        torch.Size([1, 128, 57, 76])
        torch.Size([1, 256, 29, 38])
        torch.Size([1, 512, 15, 19])
        """
