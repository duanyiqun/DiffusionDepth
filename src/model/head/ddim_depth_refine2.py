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

@HEADS.register_module()
class DDIMDepthRefine2(BaseDepthRefine):

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
        channels_in = kwargs['in_channels'][0] + self.depth_embed_dim
        # print('channels_in numbers are {}'.format(channels_in))
        if up_scale_factor == 1:
            self.up_scale = nn.Identity()
        else:
            self.up_scale = lambda tensor: F.interpolate(tensor, scale_factor=up_scale_factor, mode='bilinear')
        self.depth_transform = DEPTH_TRANSFORM.build(depth_transform_cfg)
        self.return_indices = return_indices
        self.model = ScheduledCNNRefine(channels_in=channels_in, channels_noise=kwargs['depth_feature_dim'], )
        self.diffusion_inference_steps = inference_steps
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
        self.pipeline = CNNDDIMPipiline(self.model, self.scheduler)
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
        if self.detach_fp is not False and self.detach_fp is not None:
            if isinstance(self.detach_fp, (list, tuple, range)):
                fp = [it for it in fp]
                for i in self.detach_fp:
                    fp[i] = fp[i].detach()
            else:
                fp = [it.detach() for it in fp]

        depth_map_t = self.depth_transform.t(depth_map)
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

        refined_depth_t, = self.pipeline(
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
            shape=depth_map_t.shape[-3:],
            # shape=x.shape[-3:],
            input_args=(
                x,
                depth_map_t,
                depth_map_t,
                latent_depth_mask
            ),
            num_inference_steps=self.diffusion_inference_steps,
            return_dict=False,
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
                gt_depth=gt_depth_map,
                refine_module_inputs=(
                    x,
                    depth_map_t,
                    depth_map_t,
                    latent_depth_mask
                ),
                blur_depth_t=depth_map_t,
                weight=1.0)

        output = {'pred': refined_depth, 'pred_init': depth_map_t, 'blur_depth_t': depth_map_t ,
                'ddim_loss': ddim_loss, 
                'pred_uncertainty': None,
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
        feat = feat + self.noise_embedding(noisy_image)

        ret = self.pred(feat)

        return ret