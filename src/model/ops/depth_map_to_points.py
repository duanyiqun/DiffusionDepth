import torch 


def _create_frustum(depth_map, input_size, downsample):
    '''
    depth_map: B, N, n_depth, H, W 
    '''
    # make grid in image plane
    n_depth = depth_map.shape[2]
    device = depth_map.device
    ogfH, ogfW = input_size
    fH, fW = ogfH // int(downsample), ogfW // int(downsample)
    assert fH == depth_map.shape[3] and fW == depth_map.shape[4], (fH, fW, depth_map.shape)
    ds = depth_map        
    ds = ds.clamp(0.)
    B, N_cam, D, H, W = ds.shape
    xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(B, N_cam, D, fH, fW).to(device)
    ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(B, N_cam, D, fH, fW).to(device)

    # B, N_cam, D, H, W, 3
    frustum = torch.stack((xs, ys, ds), -1)
    return frustum


def get_geometry(frustum, rots, trans, intrins, post_rots, post_trans, offset=None):
    """Determine the (x,y,z) locations (in the ego frame)
    of the points in the point cloud.
    Returns B x N x D x H/downsample x W/downsample x 3
    """
    B, N, _ = trans.shape
    # undo post-transformation
    # B x N x D x H x W x 3
    points = frustum - post_trans.view(B, N, 1, 1, 1, 3)
    if offset is not None:
        _,D,H,W = offset.shape
        points[:,:,:,:,:,2] = points[:,:,:,:,:,2]+offset.view(B,N,D,H,W)
    points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

    # cam_to_ego
    points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                        points[:, :, :, :, :, 2:3]
                        ), 5)
    if intrins.shape[3]==4: # for KITTI
        shift = intrins[:,:,:3,3]
        points  = points - shift.view(B,N,1,1,1,3,1)
        intrins = intrins[:,:,:3,:3]
    combine = rots.matmul(torch.inverse(intrins))
    points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    points += trans.view(B, N, 1, 1, 1, 3)

    # points_numpy = points.detach().cpu().numpy()
    return points


def _decorate_points(points_map, decoration_img): 
    bs, n_cam, n_depth, h, w, _ = points_map.shape 
    decoration_img = decoration_img.permute(0, 1, 3, 4, 2).unsqueeze(2) # shape: bs, n_cam, 1, h, w, 3
    decoration_img = decoration_img.expand((bs, n_cam, n_depth, h, w, 3))
    points_map = torch.cat((points_map, decoration_img), dim=-1) # shape: bs, n_cam, n_depth, h, w, 6. dimensions: x, y, z, r, g, b
    return points_map 

def convert_depth_map_to_points(depth, input_size, downsample, rots, trans, intrins, post_rots, post_trans, decoration_img=None, return_batch_idx=True):
    bs, n_cam, n_depth, h, w = depth.shape
    # depth.shape == (bs, n_cam, n_depth, h, w)
    frustum = _create_frustum(depth, input_size, downsample)
    geom = get_geometry(frustum, rots, trans, intrins, post_rots, post_trans)

    if decoration_img is not None: 
        geom = _decorate_points(geom, decoration_img)
    
    if return_batch_idx:
        geom = geom.view(-1, geom.shape[-1])

        batch_ix = torch.cat([torch.full([geom.shape[0] // bs, 1], ix,
                                            device=depth.device, dtype=torch.long) for ix in range(bs)])
        return geom, batch_ix
    else: 
        geom = geom.view(bs, -1, geom.shape[-1])
        return geom 

