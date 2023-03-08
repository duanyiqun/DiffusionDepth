
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
    

def color_depth(depth, vmin=0, vmax=200):
    

    # depth = data / 256.0
    # print(np.unique(depth))
    # assert np.min(depth) > 1, print('review depth 的 range 范围！！！')

    depth_l = np.log(depth + 3)
    # depth range [3.0m 115.0m]
    depth_near = 1
    depth_far = 115
    # log depth range [log(3.0) log(115.0)]
    # and reverse it, colormap jet [near-red far-blue]
    normalizer = mpl.colors.Normalize(vmin=-np.log(depth_far), vmax=-np.log(depth_near))  # reverse log depth range
    mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
    depth_color = (mapper.to_rgba(-depth_l)[:, :, :3] * 255).astype(np.uint8)  # reverse log depth

    # color
    # normalizer = mpl.colors.Normalize(vmin=np.min(depth), vmax=np.max(depth))
    # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    # depth_color = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)

    # tensorboard preprocess
    # depth_color = depth_color.transpose((2, 0, 1))

    return depth_color
