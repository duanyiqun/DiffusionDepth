# Copyright (c) Phigent Robotics. All rights reserved.

import os
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import numpy as np

from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.core import bbox3d2result
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.models import BACKBONES
from mmdet3d.models.builder import HEADS, build_loss
import mmcv
from collections import OrderedDict
import model.ops.ip_basic as ip_basic
from model.backbone import get as get_backbone
from model.head import get as get_head



@DETECTORS.register_module()
class Diffusion_DCx4base_Model(nn.Module):
    def __init__(
            self,
            args,
            depth_backbone_cfg=dict(type='mmbev_res18'),
            depth_head_cfg=dict(type='DDIMDepthRefine4',
                                in_channels=[64, 128, 256, 512],  # resnet18
                                depth_feature_dim=16, # deep based 16 otherwise 2 for traditional
                                loss_cfgs=[
                                        dict(loss_func='l1_depth_loss', name='depth_loss', weight=0.2, pred_indices=0, gt_indices=0), 
                                        dict(loss_func='l1_depth_loss', name='blur_depth_loss', weight=0.1, pred_indices=1, gt_indices=0),
                                    ]
                                ),
            norm_cfg=dict(type='BN'),
            depth_backbone=None,
            depth_head=None,
            center_depth_head=None,
            dense_pts_voxel_encoder=None,
            bev_fusion=None,
            pts_seg_head=None,
            lidar_stream_pretrain=None,
            cam_stream_pretrain=None,
            depth_stream_pretrain=None,
            freeze_camera_branch=False,
            freeze_lidar_branch=False,
            not_freeze_pts_bbox_head=False,
            freeze_depth_branch=False,
            with_instance_mask=False,
            ip_basic=False,
            depth_keys='all',
            **kwargs
    ):
        super(Diffusion_DCx4base_Model, self).__init__()
        """
        self.depth_backbone = mmbev_resnet.ResNetForMMBEV(3, num_layer=[2, 2, 2, 2], num_channels=[64, 128, 256, 512],
                                                          stride=[2, 2, 2, 2],
                                                          backbone_output_ids=None, norm_cfg=dict(type='BN'),
                                                          with_cp=False, block_type='Basic', )
        """
        self.args = args
        if depth_backbone is None:
            # self.depth_backbone = BACKBONES.build(depth_backbone_cfg)
            self.depth_backbone = get_backbone(args)
            self.depth_backbone = self.depth_backbone()
        if depth_head is None:
            self.depth_head = HEADS.build(depth_head_cfg)
        self.with_instance_mask = with_instance_mask
        self.ip_basic = ip_basic
        self.depth_keys = depth_keys

    def _extract_depth_ipbasic(self, img, depth_map, depth_mask, img_metas):
        B, C, imH, imW = img.shape
        depth_map = depth_map * depth_mask
        depth_map.clamp_(0, 100)
        depth_map = depth_map.view(B , *depth_map.shape[-2:]).cpu()
        ret = []
        for i in range(B ):
            depth_map_i = depth_map[i].numpy()
            dense_depth_map_i = ip_basic.fill_in_multiscale(depth_map_i)
            # dense_depth_map_i = ip_basic.fill_in_fast(depth_map_i, blur_kernel_size=3)
            ret.append(dense_depth_map_i)
        dense_depth = [torch.tensor(it, device=img.device) for it in ret]
        dense_depth = torch.stack(dense_depth)
        dense_depth = dense_depth.view(B, N, -1, *depth_map.shape[-2:])
        return dense_depth

    def extract_depth(self, img, depth_map, depth_mask, gt_depth_map, return_loss, img_metas, weight_map=None,
                      instance_masks=None, **kwargs):
        if self.ip_basic:
            return self._extract_depth_ipbasic(img, depth_map, depth_mask, img_metas)
        # 用传统算法进行的深度补全
        B, C, imH, imW = img.shape
        img = img.view(B, C, imH, imW)
        depth_map = depth_map.view(B, 1, *depth_map.shape[-2:])
        gt_depth_map = gt_depth_map.view(B, 1, *depth_map.shape[-2:]) if gt_depth_map is not None else None
        weight_map = weight_map.view(B , *depth_map.shape[-2:]) if weight_map is not None else None
        instance_masks = instance_masks.view(B, 1,
                                             *instance_masks.shape[-2:]) if instance_masks is not None else None
        depth_mask = depth_mask.view(*depth_map.shape)
        fp = self.depth_backbone(img)
        ret = self.depth_head(fp, depth_map, depth_mask, gt_depth_map=gt_depth_map, return_loss=return_loss,
                              weight_map=weight_map, instance_masks=instance_masks,
                              image=img,
                              **kwargs)
        # if return_loss: 
            # return ret
        # depth = ret.view(B, -1, *ret.shape[-2:])
        # import pdb; pdb.set_trace();
        return ret

    def forward_extra(
            self,
            points=None,
            img_metas=None,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            gt_labels=None,
            gt_bboxes=None,
            gt_masks_bev=None,
            img_inputs=None,
            depth_map=None,
            depth_mask=None,
            gt_depth_map=None,
            weight_map=None,
            proposals=None,
            gt_bboxes_ignore=None,
            instance_masks=None,
            foreground_masks=None,
            return_loss=False,
            **kwargs
    ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        loss_dict = self.extract_depth(img_inputs[0], depth_map, depth_mask, gt_depth_map, img_metas=img_metas,
                                       return_loss=False, weight_map=weight_map, instance_masks=instance_masks)
        return loss_dict

    def forward(self, sample):
        """Forward training function.
        Args:
            sample containing four keys:

            for key in sample.keys():
                print('key {}'.format(key))
                print(sample[key].shape)
            key rgb
            torch.Size([3, 3, 228, 304])
            key dep
            torch.Size([3, 1, 228, 304])
            key gt
            torch.Size([3, 1, 228, 304])
            key K
            torch.Size([3, 4])

            depth_maps = []
            for sparse_map in sparse_depth: 
                depth_map = np.asarray(sparse_map, dtype=np.float32)
                depth_map, _ = simple_depth_completion(depth_map)
                depth_maps.append(depth_map)
            depth_maps = np.stack(depth_maps)  # bs, h, w
                
        Returns:
            dict: Losses of different branches.
        """
        img_inputs = sample['rgb']
        gt_depth_map = sample['gt']
        sparse_depth = sample['dep']
        # here >=0 denotes actually we do not apply depth mask if >0 we apply 
        depth_map = sample['depth_map'] 
        depth_mask = sample['depth_mask'] 
        # print(depth_mask)
        # print(depth_map)

        output_dict = self.extract_depth(img_inputs, depth_map, depth_mask, gt_depth_map, img_metas=None,
                                       return_loss=True, weight_map=None, instance_masks=None)
        return output_dict
