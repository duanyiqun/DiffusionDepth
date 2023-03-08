import torch 
from torch import Tensor 
from typing import Union

@torch.no_grad()
def project_lidar_to_cam(
    pts, rots, trans, intrins, post_trans, post_rots, 
    height: Union[float, int, Tensor], 
    weight: Union[float, int, Tensor], 
    max_depth: float=1e9
):
    n_cam = rots.shape[0]
    n_pts = pts.shape[0]
    rots = torch.linalg.inv(rots)
    assert rots.shape == (n_cam, 3, 3)
    trans = -(rots @ trans.unsqueeze(-1)) 
    assert trans.shape == (n_cam, 3, 1)

    pts = pts[..., :3]  # n_pts, 3
    pts = rots @ pts[..., None, :, None]
    pts = pts + trans  # n_pts, n_cam, 3, 1
    depth = pts[..., 2, 0]  # n_pts, n_cam
    pts = intrins @ pts # n_pts, n_cam, 3, 1 
    pts = pts[..., 0] # n_pts, n_cam, 3

    list_uvdepth = []
    for i in range(n_cam):
        if isinstance(height, (float, int)): 
            h, w = height, weight 
        else: 
            h, w = height[i], weight[i]
        pts_cam_i = pts[..., :3]  # n_pts, 3
        assert pts_cam_i.shape == (n_pts, 3)
        pts_cam_i = rots[i] @ pts_cam_i[..., :, None]  # n_pts, 3, 1
        assert pts_cam_i.shape == (n_pts, 3, 1)
        pts_cam_i = pts_cam_i + trans[i]  # n_pts, 3, 1 
        assert pts_cam_i.shape == (n_pts, 3, 1)

        pts_cam_i = pts_cam_i.unsqueeze(-1)  # n_pts, 3
        depth_cam_i = pts_cam_i[..., 2]  # n_pts

        pts_cam_i = pts_cam_i[..., :2] / pts_cam_i[..., [2]]  # n_pts, 2
        isnotnan = (
            torch.all(~torch.isnan(pts_cam_i), axis=-1) & 
            torch.all(~torch.isinf(pts_cam_i), axis=-1)
        )
        pts_cam_i = pts_cam_i[isnotnan]
        depth_cam_i = depth_cam_i[isnotnan]
        pts_cam_i = pts_cam_i @ post_rots[:2, :2].T + post_trans[:2]

        mask2 = (
            (depth_cam_i > 0) & 
            (depth_cam_i <= max_depth) &
            (pts_cam_i[..., 0] >= 0) & 
            (pts_cam_i[..., 0] < w) & 
            (pts_cam_i[..., 1] >= 0) & 
            (pts_cam_i[..., 1] < h)
        )
        pts_cam_i = pts_cam_i[mask2]
        depth_cam_i = depth_cam_i[mask2]

        uvdepth = torch.cat((pts, depth[:, None]), axis=-1)  # n_pts, 3
        list_uvdepth.append(uvdepth)
    return list_uvdepth