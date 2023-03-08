"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    L1 loss implementation
"""


import torch
import torch.nn as nn


class DDIMLoss(nn.Module):
    def __init__(self, args):
        super(DDIMLoss, self).__init__()

        self.args = args
        self.t_valid = 0.0001

    def forward(self, refine_module_inputs, gt_depth, blur_depth_t, weight, **kwargs):
        noise = torch.randn(gt_depth.shape).to(gt_depth.device)
        bs = gt_depth.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_depth.device).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(blur_depth_t, noise, timesteps)

        noise_pred = self.model(noisy_images, timesteps, *refine_module_inputs)

        loss = F.mse_loss(noise_pred, noise)

        return loss