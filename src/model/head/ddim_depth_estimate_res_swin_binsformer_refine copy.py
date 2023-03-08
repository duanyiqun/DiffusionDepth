# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
from torch import nn
import torch.nn.functional as F
from mmdet3d.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule, ModuleList, force_fp32
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer, build_upsample_layer
from model.diffusers.schedulers.scheduling_ddim import DDIMScheduler
from typing import Union, Dict, Tuple, Optional
from .mmbev_base_depth_refine import BaseDepthRefine
from model.ops.depth_transform import DEPTH_TRANSFORM
from loss.submodule.chamferloss import BinsChamferLoss
from ..utils.transformer import PureMSDEnTransformer, PixelTransformerDecoder
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding, build_attention
import torch.nn.functional as F
import numpy as np
from model.ops.resize import resize


from loss.submodule.l1loss import L1Loss
from loss.submodule.l2loss import L2Loss

@HEADS.register_module()
class DDIMDepthEstimate_Swin_Binsformer_Refine(BaseDepthRefine):

    def __init__(
            self,
            up_scale_factor=1,
            inference_steps=20,
            num_train_timesteps=1000,
            return_indices=None,
            depth_transform_cfg=dict(type='DeepDepthTransformWithUpsampling', hidden=16, eps=1e-6),
            **kwargs
    ):
        super().__init__(blur_depth_head=False, **kwargs)
        # channels_in = kwargs['in_channels'][0] + self.depth_embed_dim
        self.l1_loss = L1Loss(self.init_cfg)
        self.l2_loss = L2Loss(self.init_cfg)
        fpn_dim = 512
        channels_in = fpn_dim
        multi_scale_depth_layer = 9 - 1 
        # abondon with lowest resolution
        # print('channels_in numbers are {}'.format(channels_in))
        in_channels=[192, 384, 768, 1536]
        self.in_channels = in_channels
        n_bins = 64
        self.n_bins = n_bins
        binsformer = True
        with_loss_chamfer = self.init_cfg.with_loss_chamfer
        self.max_depth = self.init_cfg.max_depth
        self.min_depth = self.init_cfg.min_depth
        self.classify = False
        """
        for siwn large 
        torch.Size([1, 192, 57, 76])
        torch.Size([1, 384, 29, 38])
        torch.Size([1, 768, 15, 19])
        torch.Size([1, 1536, 8, 10])
        """
        if up_scale_factor == 1:
            self.up_scale = nn.Identity()
        else:
            self.up_scale = lambda tensor: F.interpolate(tensor, scale_factor=up_scale_factor, mode='bilinear')
        self.depth_transform = DEPTH_TRANSFORM.build(depth_transform_cfg)
        self.return_indices = return_indices
        self.model = ScheduledCNNRefine(channels_in=multi_scale_depth_layer, channels_noise=kwargs['depth_feature_dim'], )
        self.diffusion_inference_steps = inference_steps
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
        self.pipeline = CNNDDIMPipiline(self.model, self.scheduler)
        """
        self.convup_fp = nn.Sequential(
                        build_upsample_layer(
                            cfg=dict(type='deconv', bias=False),
                            in_channels=channels_in,
                            out_channels=channels_in,
                            kernel_size=2,
                            stride=2,
                        ),
                        build_norm_layer(dict(type='BN'), channels_in)[1],
                        nn.ReLU(True),
                    )
        """
        del self.weight_head
        del self.conv_lateral
        del self.conv_up
        # del self.blur_depth_head

        conv_dim = 512
        self.conv_dim = conv_dim
        act_cfg = dict(type='LeakyReLU', inplace=True)
        self.act_cfg = act_cfg
        norm_cfg = dict(type='BN', requires_grad=True)
        self.norm_cfg = norm_cfg
        self.align_corners = True
        encoder_cfg =dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', 
                        embed_dims=512, 
                        num_levels=3, 
                        num_points=8),
                    ffn_cfgs=dict(
                        embed_dims=512,
                        feedforward_channels=1024,
                        ffn_dropout=0.1,),
                    # feedforward_channels=1024,
                    # ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')))
        self.transformer_encoder = PureMSDEnTransformer(num_feature_levels=3, encoder=encoder_cfg)

        self.positional_encoding = build_positional_encoding(dict(
                                    type='SinePositionalEncoding', num_feats=256, normalize=True))
        self.index = [0,1,2,3,4]
        self.trans_index = [1,2,3]
        self.transformer_num_feature_levels = len(self.trans_index)
        self.level_embed = nn.Embedding(self.transformer_num_feature_levels, conv_dim)
        self.with_loss_chamfer = with_loss_chamfer
        if with_loss_chamfer:
            self.loss_chamfer = BinsChamferLoss(loss_weight=1.0)
        train_cfg=dict(
        aux_loss = True,
        aux_index = [2, 5, 8],
        aux_weight = [1/4, 1/2, 1]
        )   
        self.train_cfg = train_cfg
        dms_decoder = True
        self.dms_decoder = dms_decoder

        # DMSTransformer used to apply self-att before cross-att following detr-like methods
        self.skip_proj = nn.ModuleList()
        trans_channels = [self.in_channels[i] for i in self.trans_index]
        for trans_channel in trans_channels:
            self.skip_proj.append(
                ConvModule(trans_channel,
                           self.conv_dim,
                           kernel_size=1,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg))

        # pixel-wise decoder (FPN)
        self.num_fpn_levels = len(self.trans_index)
        lateral_convs = nn.ModuleList()
        output_convs = nn.ModuleList()

        for idx, in_channel in enumerate(self.in_channels[:self.num_fpn_levels]):
            lateral_conv = ConvModule(
                in_channel, 
                conv_dim, 
                kernel_size=1, 
                norm_cfg=norm_cfg)
            output_conv = ConvModule(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        # transformer_decoder['decoder']['classify'] = False
        # transformer_decoder['classify'] = False
        # learnable query features
        self.query_feat = nn.Embedding(self.n_bins, conv_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(self.n_bins, conv_dim)

        transformerlayers_cfg=dict(
            type='PixelTransformerDecoderLayer',
            attn_cfgs=dict(
                type='MultiheadAttention',
                embed_dims=512,
                num_heads=8,
                dropout=0.0),
                ffn_cfgs=dict(
                feedforward_channels=2048,
                ffn_drop=0.0),
            operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm'))
            
        self.transformer_decoder = PixelTransformerDecoder(return_intermediate=True,
                                                            num_layers=9,
                                                            num_feature_levels=3,
                                                            hidden_dim=512,
                                                            operation='//',
                                                            classify=False,
                                                            transformerlayers=transformerlayers_cfg)

        # regression baseline
        self.binsformer = binsformer
        if binsformer is False:
            self.pred_depth = ConvModule(
                self.n_bins,
                1,
                kernel_size=3,
                stride=1,
                padding=1)

        # used in visualization
        self.hook_identify_center = torch.nn.Identity()
        self.hook_identify_prob = torch.nn.Identity()
        self.hook_identify_depth = torch.nn.Identity()
        """
        self.fuse_refined_depth = ConvModule(
                2,
                1,
                kernel_size=3,
                stride=1,
                padding=1)
        """

    def forward(self, fp, depth_map, depth_mask, gt_depth_map=None, return_loss=False, **kwargs):
        """
        fp: List[Tensor]
        depth_map: Tensor with shape bs, 1, h, w
        depth_mask: Tensor with shape bs, 1, h, w
        """
        if self.detach_fp is not False and self.detach_fp is not None:
            if isinstance(self.detach_fp, (list, tuple, range)):
                fp = [it for it in fp]
                for i in self.detach_fp:
                    fp[i] = fp[i].detach()
            else:
                fp = [it.detach() for it in fp]
        
        ###################################################
        # DMS Encoder and pre processing with fpn
        ###################################################
        inputs = fp
        out = []
        if self.dms_decoder:
            
            # projection of the input features
            trans_feats = [inputs[i] for i in self.trans_index]

            mlvl_feats = [
                skip_proj(trans_feats[i])
                for i, skip_proj in enumerate(self.skip_proj)
            ]

            batch_size = mlvl_feats[0].size(0)
            input_img_h, input_img_w = mlvl_feats[0].size(2), mlvl_feats[0].size(3)
            img_masks = mlvl_feats[0].new_zeros(
                (batch_size, input_img_h, input_img_w))

            mlvl_masks = []
            mlvl_positional_encodings = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(img_masks[None],
                                size=feat.shape[-2:]).to(torch.bool).squeeze(0))
                mlvl_positional_encodings.append(
                    self.positional_encoding(mlvl_masks[-1]))

            
            feats = self.transformer_encoder(mlvl_feats, mlvl_masks, mlvl_positional_encodings)
            
            split_size_or_sections = [None] * self.transformer_num_feature_levels

            for i in range(self.transformer_num_feature_levels):
                bs, _, h, w = mlvl_feats[i].shape
                split_size_or_sections[i] = h * w
                
            y = torch.split(feats, split_size_or_sections, dim=1)

            for i, z in enumerate(y):
                out.append(z.transpose(1, 2).view(bs, -1, mlvl_feats[i].size(2), mlvl_feats[i].size(3)))

            out = out[::-1]

        # NOTE: pixel-wise decoder to obtain the hr feature map
        multi_scale_features = []
        num_cur_levels = 0
        # append `out` with extra FPN levels (following MaskFormer, Mask2Former)
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.index[:self.num_fpn_levels][::-1]):

            x = inputs[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)

            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=True)
            y = output_conv(y)

            out.append(y)
        
        # the features in list out:
        # binsformer out = [1/32 enh feat, 1/16 enh feat, 1/8 enh feat, ... 
        #                   **DMSTransformer output(or naive inputs), low res to high res.**
        #                   **totally have self.transformer_num_feature_levels feats witch will interact with the bins queries**
        #                   temp feat, temp feat, temp feat, ..., per-pixel final-feat]

        for o in out:
            if num_cur_levels < self.transformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        
        # NOTE: transformer decoder
        per_pixel_feat = out[-1]
        pred_bins = []
        pred_depths = []
        pred_classes = []
        ###################################################
        # DMS Encoder and pre processing with fpn
        ###################################################
        # for o in out:
        #     print("output sequence {}".format(o.shape))
        # print('pixel wise model outputs {}'.format(per_pixel_feat.shape))

        ###################################################
        # Bins decoder
        ###################################################

        # deal with multi-scale feats
        mlvl_feats = multi_scale_features

        src = []
        pos = []
        size_list = []

        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = mlvl_feats[0].size(2), mlvl_feats[0].size(3)
        img_masks = mlvl_feats[0].new_zeros(
            (batch_size, input_img_h, input_img_w))

        mlvl_masks = []
        for idx, feat in enumerate(mlvl_feats):
            size_list.append(feat.shape[-2:])
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                            size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            pos.append(
                self.positional_encoding(mlvl_masks[-1]).flatten(2) + self.level_embed.weight[idx][None, :, None])
            src.append(feat.flatten(2))

            # 4, 256, 14144 -> HW, N, C
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
        
        multi_scale_infos = {'src':src, 'pos':pos, 'size_list':size_list}

        bs = per_pixel_feat.shape[0]
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        query_pe = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_bins, predictions_logits, predictions_class = \
             self.transformer_decoder(multi_scale_infos, query_feat, query_pe, per_pixel_feat)
        # print('raw inputs this predictions_bins {}'.format(predictions_bins.shape))
        # NOTE: depth estimation module
        self.norm = 'softmax'
        # cont = 0 
        for item_bin, pred_logit, pred_class in \
            zip(predictions_bins, predictions_logits, predictions_class):
            
            if self.binsformer is False:
                pred_depth = F.relu(self.pred_depth(pred_logit)) + self.min_depth

            else:

                bins = item_bin.squeeze(dim=2)
                
                if self.norm == 'linear':
                    bins = torch.relu(bins)
                    eps = 0.1
                    bins = bins + eps
                elif self.norm == 'softmax':
                    bins = torch.softmax(bins, dim=1)
                else:
                    bins = torch.sigmoid(bins)
                bins = bins / bins.sum(dim=1, keepdim=True)

                bin_widths = (self.max_depth - self.min_depth) * bins  # .shape = N, dim_out
                bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
                bin_edges = torch.cumsum(bin_widths, dim=1)
                centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
                n, dout = centers.size()
                centers = centers.contiguous().view(n, dout, 1, 1)

                pred_logit = pred_logit.softmax(dim=1)
                pred_depth = torch.sum(pred_logit * centers, dim=1, keepdim=True)

                centers = self.hook_identify_center(centers)
                pred_logit = self.hook_identify_prob(pred_logit)
            
                pred_bins.append(bin_edges)
                pred_classes.append(pred_class) 
                # print("bins edges is {}".format(bin_edges.shape))
                # print("pred_depth is {}".format(pred_depth.shape))
                # cont=cont + 1
                # print("depth dim is {}".format(cont))
            pred_depths.append(pred_depth)

        # pred_depths, pred_bins, pred_classes

        ###################################################
        # Bins decoder
        ###################################################


        ###################################################
        # compute losses
        losses = dict()

        aux_weight_dict = {}

        depth_gt = gt_depth_map

        if self.train_cfg["aux_loss"]:

            for index, weight in zip(self.train_cfg["aux_index"], self.train_cfg["aux_weight"]):
                depth = pred_depths[index]

                depth = resize(
                    input=depth,
                    size=depth_gt.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
                
                if self.binsformer is False:
                    depth_loss = self.loss_decode(depth, depth_gt) * weight

                else:
                    depth_loss = self.loss_decode(depth, depth_gt) * weight

                    if self.classify:
                        cls = pred_classes[index]
                        loss_ce, _ = self.loss_class(cls, class_label)
                        aux_weight_dict.update({'aux_loss_ce' + f"_{index}": loss_ce})

                    if self.with_loss_chamfer:
                        bin = pred_bins[index]
                        bins_loss = self.loss_chamfer(bin, depth_gt) * weight
                        aux_weight_dict.update({'aux_loss_chamfer' + f"_{index}": bins_loss})
                
                aux_weight_dict.update({'aux_loss_depth' + f"_{index}": depth_loss})
            
            losses.update(aux_weight_dict)


         # main loss
        """
        depth = pred_depths[-1]
        
        depth = resize(
            input=depth,
            size=depth_gt.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)

        # depth = self.attention_a * refined_depth + self.attention_b * depth 

        if self.binsformer is False:
            depth_loss = self.loss_decode(depth, depth_gt)
        else:
            depth_loss = self.loss_decode(depth, depth_gt)

            if self.classify:
                cls = pred_classes[-1]
                loss_ce, acc = self.loss_class(cls, class_label) 
                losses["loss_ce"] = loss_ce
                for index, topk in enumerate(acc):
                    losses["ce_acc_level_{}".format(index)] = topk

            if self.with_loss_chamfer:
                bin = pred_bins[-1]
                bins_loss = self.loss_chamfer(bin, depth_gt)
                losses["loss_chamfer"] = bins_loss

        losses["loss_depth"] = depth_loss
        """

        discrete_blur_depth = torch.cat(pred_depths, dim=1)
        ###################################################
        # DDIM Pipline receives pixel wise feature
        ###################################################
        # depth_map_t = self.depth_transform.t(depth_map)
        # x = discrete_blur_depth
        x = discrete_blur_depth[:, :8, :, :]
        # trash
        gt_map_t = self.depth_transform.t(gt_depth_map)
        refined_depth_t, = self.pipeline(
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
            shape=gt_map_t.shape[-3:],
            # shape=x.shape[-3:],
            input_args=(
                x,
                None,
                None,
                None
            ),
            num_inference_steps=self.diffusion_inference_steps,
            return_dict=False,
        )
        # print('final_latent_output {}'.format(refined_depth_t.shape))
        refined_depth = self.depth_transform.inv_t(refined_depth_t)
        
        
        bin_depth = resize(
            input=pred_depths[-1],
            size=depth_gt.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        
        
        # refined_depth = self.fuse_refined_depth(torch.cat((refined_depth, bin_depth), dim=1))
        
        """
        if return_loss:
            return self.loss(
                pred_depth=refined_depth,
                gt_depth=gt_depth_map,
                refine_module_inputs=(
                    x,
                    depth_map_t,
                    depth_map_t,
                    latent_depth_mask
                ),
                blur_depth_t=depth_map_t,
                **kwargs
            )
        """
        ddim_loss = self.ddim_loss(
                pred_depth=refined_depth,
                gt_depth=gt_map_t,
                refine_module_inputs=(
                    x,
                    None,
                    None,
                    None
                ),
                blur_depth_t=refined_depth_t,
                weight=1.0)
        


        ###################################################
        # DDIM Pipline receives pixel wise feature
        ###################################################

    
        output = {'pred': refined_depth, 'pred_init': bin_depth, 'blur_depth_t': gt_map_t ,
                'ddim_loss': ddim_loss, 'gt_map_t': gt_map_t, 
                'pred_uncertainty': None, 'bin_losses': losses,
                 'pred_inter': None, 'weight_map': None,
                  'guidance': None, 'offset': None, 'aff': None,
                  'gamma': None, 'confidence': None}


        return output

    def loss(self, pred_depth, gt_depth, refine_module_inputs, blur_depth_t, pred_uncertainty=None, weight_map=None,
             **kwargs):
        loss_dict = super().loss(pred_depth, gt_depth, pred_uncertainty, weight_map, **kwargs)
        for loss_cfg in self.loss_cfgs:
            loss_fnc_name = loss_cfg['loss_func']
            loss_key = loss_cfg['name']
            if loss_key == 'ddim_loss':
                loss_fnc = self.ddim_loss
            else:
                continue
            loss = loss_fnc(
                pred_depth=pred_depth, pred_uncertainty=pred_uncertainty,
                gt_depth=gt_depth,
                refine_module_inputs=refine_module_inputs,
                blur_depth_t=blur_depth_t,
                weight_map=weight_map, **loss_cfg, **kwargs
            )
            loss_dict[loss_key] = loss
        return loss_dict

    def ddim_loss(self, gt_depth, refine_module_inputs, blur_depth_t, weight, **kwargs):
        # Sample noise to add to the images
        noise = torch.randn(blur_depth_t.shape).to(blur_depth_t.device)
        bs = blur_depth_t.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_depth.device).long()
        # 这里的随机是在 bs维度，这个情况不能太小。
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(blur_depth_t, noise, timesteps)

        noise_pred = self.model(noisy_images, timesteps, *refine_module_inputs)

        loss = F.mse_loss(noise_pred, noise)

        return loss

    def ddim_loss_gt(self, gt_depth, refine_module_inputs, blur_depth_t, weight, **kwargs):
        # Sample noise to add to the images
        noise = torch.randn(gt_depth.shape).to(gt_depth.device)
        bs = gt_depth.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_depth.device).long()
        # 这里的随机是在 bs维度，这个情况不能太小。
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_depth, noise, timesteps)

        noise_pred = self.model(noisy_images, timesteps, *refine_module_inputs)

        loss = F.mse_loss(noise_pred, noise)

        return loss
    
    def loss_decode(self, depth_predict, depth_gt, weightl1=0.5, weightl2=0.5):
        loss_decode = weightl1 * self.l1_loss(depth_predict, depth_gt) + weightl2 * self.l2_loss(depth_predict, depth_gt)
        return loss_decode
        
        


class CNNDDIMPipiline:
    '''
    Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py
    '''

    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    def __call__(
            self,
            batch_size,
            device,
            dtype,
            shape,
            input_args,
            generator: Optional[torch.Generator] = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            return_dict: bool = True,
            **kwargs,
    ) -> Union[Dict, Tuple]:
        if generator is not None and generator.device.type != self.device.type and self.device.type != "mps":
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{self.device}`, so the `generator` will be ignored. "
                f'Please use `generator=torch.Generator(device="{self.device}")` instead.'
            )
            raise RuntimeError(
                "generator.device == 'cpu'",
                "0.11.0",
                message,
            )
            generator = None

        # Sample gaussian noise to begin loop
        image_shape = (batch_size, *shape)

        image = torch.randn(image_shape, generator=generator, device=device, dtype=dtype)
        # print('random_noise is {}'.format(image.shape))
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            # timesteps 选择了20步
            # 1. predict noise model_output
            model_output = self.model(image, t.to(device), *input_args)

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=True, generator=generator
            )['prev_sample']

        if not return_dict:
            return (image,)

        return {'images': image}


class UpSample(nn.Sequential):
    '''Fusion module
    From Adabins
    
    '''
    def __init__(self, skip_input, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(UpSample, self).__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)))


class UpSample_add(nn.Sequential):
    '''Fusion module
    From Adabins
    
    '''
    def __init__(self, skip_input, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(UpSample_add, self).__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.convB(self.convA(up_x + concat_with))


class ScheduledCNNRefine(BaseModule):
    def __init__(self, channels_in, channels_noise, **kwargs):
        super().__init__(**kwargs)
        self.noise_embedding = nn.Sequential(
            nn.Conv2d(channels_noise, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 64),
            # 不能用batch norm，会统计输入方差，方差会不停的变
            nn.ReLU(True),
            nn.Conv2d(64, channels_in, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, channels_in),
            nn.ReLU(True),
        )
        self.upsample_fuse = UpSample_add(channels_in, channels_in)

        self.time_embedding = nn.Embedding(1280, channels_in)

        self.pred = nn.Sequential(
            nn.Conv2d(channels_in, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(True),
            nn.Conv2d(64, channels_noise, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, channels_noise),
            nn.ReLU(True),
        )

    def forward(self, noisy_image, t, *args):
        feat, blur_depth, sparse_depth, sparse_mask = args
        # print('debug: feat shape {}'.format(feat.shape))
        # diff = (noisy_image - blur_depth).abs()
        if t.numel() == 1:
            # print(t)
            feat = feat + self.time_embedding(t)[..., None, None]
            # feat = feat + self.time_embedding(t)[None, :, None, None]
            # t 如果本身是一个值，需要扩充第一个bs维度 (这个暂时不适用)
        else:
            # print(t)
            feat = feat + self.time_embedding(t)[..., None, None]
        # layer(feat) - noise_image
        # blur_depth = self.layer(feat); 
        # ret =  a* noisy_image - b * blur_depth
        # print('debug: noisy_image shape {}'.format(noisy_image.shape))
        # feat = feat + self.noise_embedding(noisy_image)
        feat = self.upsample_fuse(feat, self.noise_embedding(noisy_image))

        ret = self.pred(feat)

        return ret