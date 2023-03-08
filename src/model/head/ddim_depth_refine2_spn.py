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
from ..modulated_deform_conv_func import ModulatedDeformConvFunction
from ..common import conv_bn_relu

@HEADS.register_module()
class DDIMDepthRefine2_SPN(BaseDepthRefine):

    def __init__(
            self,
            up_scale_factor=1,
            inference_steps=20,
            num_train_timesteps=200,
            return_indices=None,
            depth_transform_cfg=dict(type='DeepDepthTransformWithUpsampling1x1', hidden=16, eps=1e-6),
            **kwargs
    ):
        super().__init__(blur_depth_head=False, **kwargs)
        # print(self.init_cfg)
        channels_in = kwargs['in_channels'][0] + self.depth_embed_dim
        self.num_neighbors = self.init_cfg.prop_kernel*self.init_cfg.prop_kernel - 1
        # print('channels_in numbers are {}'.format(channels_in))
        self.preserve_input = self.init_cfg.preserve_input
        if up_scale_factor == 1:
            self.up_scale = nn.Identity()
        else:
            self.up_scale = lambda tensor: F.interpolate(tensor, scale_factor=up_scale_factor, mode='bilinear')
        self.depth_transform = DEPTH_TRANSFORM.build(depth_transform_cfg)
        self.return_indices = return_indices
        self.model = ScheduledCNNRefine(channels_in=channels_in, channels_noise=kwargs['depth_feature_dim'], )
        self.diffusion_inference_steps = inference_steps
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
        self.pipeline = SPNDDIMPipiline(self.init_cfg, self.model, self.scheduler, 
                                        ch_g=self.num_neighbors, ch_f=kwargs['depth_feature_dim'], 
                                        k_g=3, k_f = self.init_cfg.prop_kernel, feature_inchannel=channels_in,
                                        depth_dim=kwargs['depth_feature_dim'])
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
        del self.weight_head

    def forward(self, fp, depth_map, depth_mask, gt_depth_map=None, return_loss=False, **kwargs):
        """
        fp: List[Tensor]
        depth_map: Tensor with shape bs, 1, h, w
        depth_mask: Tensor with shape bs, 1, h, w
        """
        sparse_depth = kwargs['sparse_depth']
        if self.detach_fp is not False and self.detach_fp is not None:
            if isinstance(self.detach_fp, (list, tuple, range)):
                fp = [it for it in fp]
                for i in self.detach_fp:
                    fp[i] = fp[i].detach()
            else:
                fp = [it.detach() for it in fp]
        # print(depth_map.shape)
        depth_map_t = self.depth_transform.t(depth_map)
        # print(depth_map_t.shape)
        # 这里给GT也转换成depth_map
        gt_map_t = self.depth_transform.t(gt_depth_map)
        # print(sparse_depth)
        sparse_depth_t = self.depth_transform.t(sparse_depth)
        # print(sparse_depth_t)
        # down scale to latent 
        # 多层感知机/人为设定 很多通道怎么 变成深度值
        latent_depth_mask = nn.functional.adaptive_max_pool2d(depth_mask.float(), output_size=depth_map_t.shape[-2:])
        depth = torch.cat((depth_map_t, latent_depth_mask), dim=1)  # bs, 2, h, w if traditional bs, 1+dim, h, w if deep
        # 模型里面隐形编码了mask 哪些是真值
        for i in range(len(fp)):
            f = fp[len(fp) - i - 1]
            depth_down = nn.functional.adaptive_avg_pool2d(depth, output_size=f.shape[-2:])
            depth_embed = self.conv_lateral[len(fp) - i - 1](depth_down)
            # conv_lateral 只是通道转换
            x = torch.cat((f, depth_embed), axis=1)
            # x = f
            # print('current x {}'.format(x.shape))
            if i > 0:
                # print('current pre_x {}'.format(pre_x.shape)) # in case some odd numbers, nyudepth shape is fixed
                x = x + nn.functional.adaptive_avg_pool2d(self.conv_up[len(fp) - i - 1](pre_x), output_size=x.shape[-2:])
            pre_x = x
            # 和ddim random feature map是一样的尺寸 （长宽一样，通道数不一定）
            # x 是condition，没有参与真值回归
        # x = self.convup_fp(x)
        # upscale x into depth real size will crush the me

        refined_depth_t, feat_inter, offset, aff, aff_const, guidance, confidence = self.pipeline(
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
            shape=depth_map_t.shape[-3:],
            sparse_depth_t=sparse_depth_t,
            # shape=x.shape[-3:],
            input_args=(
                x,
                depth_map_t,
                depth_map_t,
                latent_depth_mask
            ),
            num_inference_steps=self.diffusion_inference_steps,
            return_dict=False,
            preserve_input= self.preserve_input
        )
        # print('final_latent_output {}'.format(refined_depth_t.shape))
        refined_depth = self.depth_transform.inv_t(refined_depth_t)
        
        # refine depth 直接输出了，还没有cspn这个module

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
                    depth_map_t,
                    depth_map_t,
                    latent_depth_mask
                ),
                blur_depth_t=depth_map_t,
                weight=1.0)

        output = {'pred': refined_depth, 'pred_init': depth_map_t, 'blur_depth_t': depth_map_t ,
                'ddim_loss': ddim_loss, 'gt_map_t': gt_map_t, 
                'pred_uncertainty': None,
                 'pred_inter': feat_inter, 'weight_map': None,
                  'guidance': guidance, 'offset': offset, 'aff': aff,
                  'gamma': aff_const, 'confidence': F.interpolate(confidence, scale_factor=2)}


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


class SPNDDIMPipiline(BaseModule):
    '''
    Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py
    '''

    def __init__(self, args, model, scheduler, ch_g, ch_f, k_g, k_f, feature_inchannel, depth_dim):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        # assert ch_f == 16, 'only tested with ch_f == 16 but {}'.format(ch_f)
        self.args = args
        self.affinity = self.args.affinity

        assert (k_g % 2) == 1, \
            'only odd kernel is supported but k_g = {}'.format(k_g)
        pad_g = int((k_g - 1) / 2)
        assert (k_f % 2) == 1, \
            'only odd kernel is supported but k_f = {}'.format(k_f)
        pad_f = int((k_f - 1) / 2)

        self.ch_g = ch_g
        self.num_neighbors = ch_g
        # self.ch_f = ch_f
        self.ch_f = 16
        self.k_g = k_g
        self.k_f = k_f
        # Assume zero offset for center pixels
        self.num = self.k_f * self.k_f - 1
        self.idx_ref = self.num // 2

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            self.conv_offset_aff = nn.Conv2d(
                self.ch_g, 3 * self.num, kernel_size=self.k_g, stride=1,
                padding=pad_g, bias=True
            )
            self.conv_offset_aff.weight.data.zero_()
            self.conv_offset_aff.bias.data.zero_()

            if self.affinity == 'TC':
                self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            elif self.affinity == 'TGASS':
                self.aff_scale_const = nn.Parameter(
                    self.args.affinity_gamma * self.num * torch.ones(1))
            else:
                self.aff_scale_const = nn.Parameter(torch.ones(1))
                self.aff_scale_const.requires_grad = False
        else:
            raise NotImplementedError

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((self.ch_f, 1, 1, 1)))
        # self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64
        self.guidance_projection = conv_bn_relu(feature_inchannel+depth_dim, self.num_neighbors, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False)
        if self.args.conf_prop:
            # Confidence Branch
            # Confidence is shared for propagation and mask generation
            # 1/1
            self.cf_dec1 = conv_bn_relu(feature_inchannel, 32, kernel=3, stride=1,
                                        padding=1)
            self.cf_dec0 = nn.Sequential(
                nn.Conv2d(32+depth_dim, self.ch_f, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )

    def _get_offset_affinity(self, guidance, confidence=None, rgb=None):
        B, _, H, W = guidance.shape
        # print(guidance.shape)

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            offset_aff = self.conv_offset_aff(guidance)
            o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

            # Add zero reference offset
            offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
            list_offset = list(torch.chunk(offset, self.num, dim=1))
            list_offset.insert(self.idx_ref,
                               torch.zeros((B, 1, 2, H, W)).type_as(offset))
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

            if self.affinity in ['AS', 'ASS']:
                pass
            elif self.affinity == 'TC':
                aff = torch.tanh(aff) / self.aff_scale_const
            elif self.affinity == 'TGASS':
                aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Apply confidence
        # TODO : Need more efficient way
        if self.args.conf_prop:
            list_conf = []
            offset_each = torch.chunk(offset, self.num + 1, dim=1)

            modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()

            for idx_off in range(0, self.num + 1):
                ww = idx_off % self.k_f
                hh = idx_off // self.k_f

                if ww == (self.k_f - 1) / 2 and hh == (self.k_f - 1) / 2:
                    continue

                offset_tmp = offset_each[idx_off].detach()

                # NOTE : Use --legacy option ONLY for the pre-trained models
                # for ECCV20 results.
                if self.args.legacy:
                    offset_tmp[:, 0, :, :] = \
                        offset_tmp[:, 0, :, :] + hh - (self.k_f - 1) / 2
                    offset_tmp[:, 1, :, :] = \
                        offset_tmp[:, 1, :, :] + ww - (self.k_f - 1) / 2
                """
                # force to unify value for different layers
                # print(confidence.shape)
                # print(offset_tmp.shape)
                print('confidence.shape {}'.format(confidence.shape) )
                print('offset_tmp.shape {}'.format(offset_tmp.shape) )
                print('modulation_dummy.shape {}'.format(modulation_dummy.shape) )
                print('w_conf.shape {}'.format(self.w_conf.shape) )
                print('self.b.shape {}'.format(self.b.shape) )
                const int channels = input.size(1);
                const int channels_out = weight.size(0);
                这里 w_conf 是每个的weight，决定了 channels out，这里如果考虑16层深度图各自有个权重，这里需要require grad
                这里 confidence 是input, 如果是input的话，也必须是有深度图维度的输出
                print('conf_tmp.shape {}'.format(conf_tmp.shape) )
                """
                conf_tmp = ModulatedDeformConvFunction.apply(
                    confidence, offset_tmp, modulation_dummy, self.w_conf,
                    self.b, self.stride, 0, self.dilation, self.groups,
                    self.deformable_groups, self.im2col_step)
                list_conf.append(conf_tmp)

            conf_aff = torch.cat(list_conf, dim=1)
            aff = torch.repeat_interleave(aff, repeats=self.ch_f, dim=1)
            # assign 8 neighbors with 16 depth maps list conf have each neighter with dephts map times repeat aff needs repeat as well
            aff = aff * conf_aff.contiguous()
            

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        if self.affinity in ['ASS', 'TGASS']:
            aff_abs_sum[aff_abs_sum < 1.0] = 1.0

        if self.affinity in ['AS', 'ASS', 'TGASS']:
            aff = aff / aff_abs_sum

        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum

        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        return offset, aff

    def _propagate_once(self, feat, offset, aff):
        feat = ModulatedDeformConvFunction.apply(
            feat, offset, aff, self.w, self.b, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups, self.im2col_step
        )

        return feat

    def forward(
            self,
            batch_size,
            device,
            dtype,
            shape,
            sparse_depth_t,
            input_args,
            generator: Optional[torch.Generator] = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            return_dict: bool = True,
            preserve_input: bool = True,
            vis_ddpm: bool = False,
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
        feat, simple_depth_map_t, simple_depth_map_t, sparse_mask = input_args

        # calculate guidance and confidance 
        # sparse_depth_t = sparse_depth_t.to(feat.device)
        guidance = self.guidance_projection(torch.cat((feat, sparse_depth_t), dim=1))

        if self.args.conf_prop:
            # Confidence Decoding
            cf_fd1 = self.cf_dec1(feat)
            confidence = self.cf_dec0(torch.cat((cf_fd1, sparse_depth_t), dim=1))
        else:
            confidence = None
        
        # calculate guidance and confidance 
        if self.args.conf_prop:
            offset, aff = self._get_offset_affinity(guidance, confidence, rgb=None)
        else:
            offset, aff = self._get_offset_affinity(guidance, None, rgb=None)

        # print("offset {}".format(offset.shape))
        # print("aff {}".format(aff.shape))
        # Sample gaussian noise to begin loop
        image_shape = (batch_size, *shape)

        image = torch.randn(image_shape, generator=generator, device=device, dtype=dtype)
        # print('random_noise is {}'.format(image.shape))
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        if preserve_input: 
            assert image.shape[2:] == sparse_depth_t.shape[2:]
            mask_fix = torch.sum(sparse_depth_t > 0.0, dim=1, keepdim=True).detach()
            mask_fix = (mask_fix > 0.0).type_as(image)

        list_feat = []

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
            # sparse_depth 
            # print('mask_shape {}'.format(sparse_mask.shape))
            # print('sparse_depth_t {}'.format(sparse_depth_t.shape))
            # print(sparse_mask)
            # print(torch.sum(sparse_mask))
            if preserve_input: 
                image = (1.0 - mask_fix) * image \
                                + mask_fix * sparse_depth_t
            
            image = self._propagate_once(image, offset, aff)
            if vis_ddpm:
                list_feat.append(image)
            else:
                list_feat = None
            

        if not return_dict:
            return (image, list_feat, offset, aff, self.aff_scale_const.data, guidance, confidence )

        return {'images': image, 'list_feat':list_feat, 'offset':offset, 
                'aff': aff, 'aff_consta': self.aff_scale_const.data, 
                'guidance':guidance, 'confidence':confidence }


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
        # feat = feat + self.noise_embedding(noisy_image,feat)
        feat = feat + self.noise_embedding(noisy_image)
        ret = self.pred(feat)

        return ret