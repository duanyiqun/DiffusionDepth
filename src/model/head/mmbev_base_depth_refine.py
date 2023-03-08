# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, ModuleList, force_fp32
from torch import nn
import torch.nn.functional as F
from model.ops.depth_map_to_points import convert_depth_map_to_points

from model.ops.depth_transform import DEPTH_TRANSFORM


class BaseDepthRefine(BaseModule):
    def __init__(
            self,
            in_channels,
            detach_fp=False,
            blur_depth_head=True,
            depth_embed_dim=16,
            depth_feature_dim=16,
            loss_cfgs=[],
            depth_transform_cfg=dict(type='ReciprocalDepthTransform'),
            upsample_cfg=dict(type='deconv', bias=False),
            norm_cfg=dict(type='BN'),
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        depth_feature_dim=16
        self.init_cfg = init_cfg
        self.detach_fp = detach_fp
        self.depth_embed_dim = depth_embed_dim
        self.depth_transform = DEPTH_TRANSFORM.build(depth_transform_cfg)
        self.loss_cfgs = loss_cfgs
        self.conv_lateral = ModuleList()
        self.conv_up = ModuleList()
        for i in range(len(in_channels)):
            self.conv_lateral.append(
                nn.Sequential(
                    nn.Conv2d(depth_feature_dim + 1, depth_embed_dim, 3, 1, 1, bias=False),
                    build_norm_layer(norm_cfg, depth_embed_dim)[1],
                    nn.ReLU(True),
                    #     nn.Conv2d(depth_embed_dim, depth_embed_dim, 3, 1, 1, bias=False),
                    #     build_norm_layer(norm_cfg, depth_embed_dim)[1],
                    #     nn.ReLU(True),
                )
            )

            if i != 0:
                self.conv_up.append(
                    nn.Sequential(
                        build_upsample_layer(
                            upsample_cfg,
                            in_channels=in_channels[i] + depth_embed_dim,
                            out_channels=in_channels[i - 1] + depth_embed_dim,
                            kernel_size=2,
                            stride=2,
                        ),
                        build_norm_layer(norm_cfg, in_channels[i - 1] + depth_embed_dim)[1],
                        nn.ReLU(True),
                    )
                )

        if blur_depth_head:
            self.blur_depth_head = nn.Sequential(
                nn.Conv2d(in_channels[0] + depth_embed_dim, depth_embed_dim, 3, 1, 1, bias=False),
                build_norm_layer(norm_cfg, depth_embed_dim)[1],
                nn.ReLU(True),
                nn.Conv2d(depth_embed_dim, 1, 3, 1, 1, ),
                nn.Sigmoid()
            )

        self.weight_head = nn.Sequential(
            nn.Conv2d(in_channels[0] + depth_embed_dim, depth_embed_dim, 3, 1, 1, bias=False),
            build_norm_layer(norm_cfg, depth_embed_dim)[1],
            nn.ReLU(True),
            nn.Conv2d(depth_embed_dim, 1, 3, 1, 1),
        )

    def init_weights(self):
        super().init_weights()

    def forward(self, fp, depth_map, depth_mask):
        '''
        fp: List[Tensor]
        depth_map: Tensor with shape bs, 1, h, w
        depth_mask: Tensor with shape bs, 1, h, w
        '''

        if self.detach_fp is not False and self.detach_fp is not None:
            if isinstance(self.detach_fp, (list, tuple, range)):
                fp = [it for it in fp]
                for i in self.detach_fp:
                    fp[i] = fp[i].detach()
            else:
                fp = [it.detach() for it in fp]

        depth_map_t = self.depth_transform.t(depth_map)
        depth = torch.cat((depth_map_t, depth_mask), dim=1)  # bs, 2, h, w
        for i in range(len(fp)):
            f = fp[len(fp) - i - 1]
            depth_down = nn.functional.adaptive_avg_pool2d(depth, output_size=f.shape[-2:])
            depth_embed = self.conv_lateral[len(fp) - i - 1](depth_down)
            x = torch.cat((f, depth_embed), axis=1)
            # x = f
            if i > 0:
                x = x + self.conv_up[len(fp) - i - 1](pre_x)
            pre_x = x

        if hasattr(self, 'blur_depth_head'):
            blur_depth = self.blur_depth_head(x)
            blur_depth = self.depth_transform.inv_t(blur_depth)
        else:
            blur_depth = depth_map

        depth_weight = self.weight_head(x).sigmoid().clamp(1e-3, 1 - 1e-3)
        return x, blur_depth, depth_weight

    def loss(self, pred_depth, gt_depth, pred_uncertainty=None, weight_map=None, instance_masks=None, image=None,
             **kwargs):
        loss_dict = {}
        for loss_cfg in self.loss_cfgs:
            loss_fnc_name = loss_cfg['loss_func']
            loss_key = loss_cfg['name']
            if loss_fnc_name not in depth_loss_dict:
                continue
            loss = depth_loss_dict[loss_fnc_name](
                pred_depth=pred_depth, pred_uncertainty=pred_uncertainty,
                gt_depth=gt_depth, weight_map=weight_map,
                instance_masks=instance_masks,
                image=image,
                **loss_cfg, **kwargs
            )
            loss_dict[loss_key] = loss
        return loss_dict


def l1_depth_loss(pred_depth, gt_depth, pred_indices=None, gt_indices=None, weight=1., weight_map=None, **kwargs):
    # print(pred_indices)
    # if pred_indices is not None:
    #     pred_depth = pred_depth[..., pred_indices, :, :]
    # if gt_indices is not None:
    #     gt_depth = gt_depth[..., gt_indices, :, :]
    assert gt_depth.shape == pred_depth.shape, (gt_depth.shape, pred_depth.shape)
    gt_mask = gt_depth >= 0.0001
    loss = (pred_depth - gt_depth).abs() * gt_mask
    if weight_map is not None:
        loss *= weight_map
    loss = loss.sum() / gt_mask.sum().clamp(1.)
    return loss * weight


def depth_smooth_loss(pred_depth, gt_depth, image, instance_masks, pred_indices=None, gt_indices=None, weight=1.,
                      eps=1e-6, **kwargs):
    if pred_indices is not None:
        pred_depth = pred_depth[..., pred_indices, :, :]
    if gt_indices is not None:
        gt_depth = gt_depth[..., gt_indices, :, :]
    assert gt_depth.shape == pred_depth.shape, (gt_depth.shape, pred_depth.shape)

    def try_resize(input_img, shape):
        if input_img.shape[-2:] != shape:
            old_shape = input_img.shape
            image_reshape = input_img.view(-1, 1, *old_shape[-2:])
            image_reshape = F.interpolate(image_reshape, shape)
            image_reshape = image_reshape.view(*old_shape[:-2], *shape)
            return image_reshape
        return input_img

    image = try_resize(image, pred_depth.shape[-2:])
    instance_masks = instance_masks.float()
    max_id = F.max_pool2d(instance_masks, 3, stride=1, padding=1)
    min_id = -F.max_pool2d(-instance_masks, 3, stride=1, padding=1)
    edge_masks = (max_id != min_id).float()
    edge_masks = F.adaptive_max_pool2d(edge_masks, output_size=pred_depth.shape[-2:])

    pred_depth = pred_depth * (1 - edge_masks) + pred_depth.detach() * edge_masks

    grad_depth_x = torch.abs(pred_depth[..., :-1] - pred_depth[..., 1:])
    grad_depth_y = torch.abs(pred_depth[..., :-1, :] - pred_depth[..., 1:, :])

    grad_img_x = torch.mean(torch.abs(image[..., :-1] - image[..., 1:]), -3, keepdim=False)
    grad_img_y = torch.mean(torch.abs(image[..., :-1, :] - image[..., 1:, :]), -3, keepdim=False)

    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)
    return (grad_depth_x.mean() + grad_depth_y.mean()) * weight


def shape_reg_loss(pred_depth, gt_depth, foreground_masks, gt_bboxes_3d, rots, trans, intrins, post_rots, post_trans,
                   input_size, downsample, pred_indices=None, gt_indices=None, weight=1., eps=1e-6, max_distance=1,
                   focus=False, **kwargs):
    if pred_indices is not None:
        pred_depth = pred_depth[..., pred_indices, :, :]
    if gt_indices is not None:
        gt_depth = gt_depth[..., gt_indices, :, :]
    assert gt_depth.shape == pred_depth.shape, (gt_depth.shape, pred_depth.shape)

    # 1. convert depth_map to xyz_map
    xyz, _ = convert_depth_map_to_points(pred_depth.unsqueeze(2), input_size, downsample, rots, trans, intrins,
                                         post_rots, post_trans)
    xyz = xyz.view(*pred_depth.shape, 3)

    batch_size = len(gt_bboxes_3d)
    losses = [
        _shape_reg_loss_per_batch(
            xyz[b], gt_depth[b], foreground_masks[b], gt_bboxes_3d[b],
            max_distance=max_distance
        ) for b in range(batch_size)
    ]

    if focus:
        losses = [it[it > 0] for it in losses]
    return torch.cat(losses).mean() * weight


def _shape_reg_loss_per_batch(xyz, gt_depth, foreground_masks, gt_bboxes_3d, max_distance):
    gt_boxes = gt_bboxes_3d.tensor.to(xyz.device)
    cos_theta = torch.cos(gt_boxes[:, 6])
    sin_theta = torch.sin(gt_boxes[:, 6])
    zeros = sin_theta * 0.
    ones = zeros + 1.
    rot_mat = torch.stack([
        cos_theta, -sin_theta, zeros, sin_theta, cos_theta, zeros, zeros, zeros, ones,
    ], dim=-1).view(-1, 3, 3).to(xyz.device)
    box_centers = gt_boxes[:, :3]  # n_box, 3
    box_centers[:, 2] += gt_boxes[:, 5] / 2
    box_sizes = gt_boxes[:, 3:6].to(xyz.device)  # n_box, 3

    foreground_masks = _try_resize(foreground_masks.float(), gt_depth.shape[-2:], mode='nearest')
    xyz = xyz[foreground_masks > 0.5]  # n_pts, 3
    xyz_per_box = xyz.view(-1, 1, 3) - box_centers.unsqueeze(0)  # n_pts, n, 3
    xyz_per_box = xyz_per_box.unsqueeze(-2) @ rot_mat.permute(0, 2, 1)
    xyz_per_box = xyz_per_box.squeeze(-2)  # n_pts, n_box, 3

    loss = torch.min(torch.mean(torch.relu(xyz_per_box.abs() - box_sizes), dim=-1), dim=1)[0]  # n_pts

    return loss


depth_loss_dict = {
    'l1_depth_loss': l1_depth_loss,
    'depth_smooth_loss': depth_smooth_loss,
    'shape_reg_loss': shape_reg_loss,
}


def _try_resize(input_img, shape, mode='bilinear'):
    if input_img.shape[-2:] != shape:
        old_shape = input_img.shape
        image_reshape = input_img.view(-1, 1, *old_shape[-2:])
        image_reshape = F.interpolate(image_reshape, shape, mode=mode)
        image_reshape = image_reshape.view(*old_shape[:-2], *shape)
        return image_reshape
    return input_img