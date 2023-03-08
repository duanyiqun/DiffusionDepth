from mmcv.utils import Registry
from mmcv.runner import BaseModule, ModuleList, force_fp32
from model.common import conv_bn_relu
import torch
import torch.nn as nn

DEPTH_TRANSFORM = Registry('depth_transforms', )


@DEPTH_TRANSFORM.register_module()
class DeepDepthTransformWithUpsampling(BaseModule):
    def __init__(self, hidden=16, eps=1e-6):
        # 通过clamp 来限定最大深度
        super().__init__()
        self.conv_transform = nn.Sequential(
            conv_bn_relu(1, hidden, 3, 2, 1),
            conv_bn_relu(hidden, hidden, 3, 1, 1, relu=False),
            nn.Tanh()
        )
        self.conv_inv_transform = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden, out_channels=hidden, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            conv_bn_relu(hidden, 1, 3, 1, 1, bn=False, relu=False),
            nn.Sigmoid()
        )
        self.eps = eps

    def t(self, depth):
        # 多层感知编码，channel 不为1
        return self.conv_transform(depth)

    def inv_t(self, value):
        # 多层感知解码，
        return 1.0 / self.conv_inv_transform(value).clamp(self.eps) - 1


@DEPTH_TRANSFORM.register_module()
class DeepDepthTransformWithUpsampling1x1(BaseModule):
    def __init__(self, hidden=16, eps=1e-6):
        # 通过clamp 来限定最大深度
        super().__init__()
        self.conv_transform = nn.Sequential(
            nn.Conv2d(1, hidden, 1, 1, 0, bias=False),
            nn.Conv2d(hidden, hidden, 1, 1, 0, bias=False),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv_inv_transform = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden, out_channels=hidden, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            conv_bn_relu(hidden, 1, 3, 1, 1, bn=False, relu=False),
            nn.Sigmoid()
        )
        self.eps = eps

    def t(self, depth):
        # 多层感知编码，channel 不为1
        return self.conv_transform(depth)

    def inv_t(self, value):
        # 多层感知解码，
        return 1.0 / self.conv_inv_transform(value).clamp(self.eps) - 1


@DEPTH_TRANSFORM.register_module()
class DeepDepthTransformWithUpsamplingX4(BaseModule):
    def __init__(self, hidden=16, eps=1e-6):
        # 通过clamp 来限定最大深度
        super().__init__()
        self.conv_transform = nn.Sequential(
            conv_bn_relu(1, hidden, 3, 2, 1),
            conv_bn_relu(hidden, hidden, 3, 2, 1),
            conv_bn_relu(hidden, hidden, 3, 1, 1, relu=False),
            nn.Tanh()
        )
        self.conv_inv_transform = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden, out_channels=hidden, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=hidden, out_channels=hidden, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            conv_bn_relu(hidden, 1, 3, 1, 1, bn=False, relu=False),
            nn.Sigmoid()
        )
        self.eps = eps

    def t(self, depth):
        # 多层感知编码，channel 不为1
        return self.conv_transform(depth)

    def inv_t(self, value):
        # 多层感知解码，
        return 1.0 / self.conv_inv_transform(value).clamp(self.eps) - 1


@DEPTH_TRANSFORM.register_module()
class DeepDepthTransform(BaseModule):
    def __init__(self, hidden=16, eps=1e-6):
        super().__init__()
        self.conv_transform = nn.Sequential(
            conv_bn_relu(1, hidden, 3, 1, 1),
            conv_bn_relu(hidden, hidden, 3, 1, 1, relu=False),
            nn.Tanh()
        )
        self.conv_inv_transform = nn.Sequential(
            conv_bn_relu(hidden, hidden, 3, 1, 1),
            conv_bn_relu(hidden, 1, 3, 1, 1, relu=False),
            nn.Sigmoid()
        )
        self.eps = eps

    def t(self, depth):
        return self.conv_transform(depth)

    def inv_t(self, value):
        return 1.0 / self.conv_inv_transform(value).clamp(self.eps) - 1


@DEPTH_TRANSFORM.register_module()
class ReciprocalDepthTransform(object):
    def __init__(self, linear=(1, 0), eps=1e-6):
        self.linear = linear
        self.eps = eps

    def t(self, depth):
        '''
        return: transformed depth value in range (0, 1],
        '''
        return self.linear[0] / (1 + depth.clamp(0.)).clamp(self.eps) + self.linear[1]

    def inv_t(self, value):
        return (self.linear[0] / (value - self.linear[1]).clamp(self.eps) - 1)


@DEPTH_TRANSFORM.register_module()
class ReciprocalDepthTransformII(object):
    def __init__(self, min_depth=0.5):
        self.min_depth = min_depth

    def t(self, depth):
        return self.min_depth / depth.clamp(self.min_depth)

    def inv_t(self, value):
        return self.min_depth / value

