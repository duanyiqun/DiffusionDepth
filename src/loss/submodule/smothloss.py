import torch
import torch.nn as nn


class SmoothLoss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()

        self.args = args
        self.t_valid = 0.0001

    def forward(self, pred_depth, gt_depth, image, instance_masks, pred_indices=None, gt_indices=None, weight=1.,
                      eps=1e-6):
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
