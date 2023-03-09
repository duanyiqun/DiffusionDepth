from importlib import import_module
from .ddim_depth_estimate_res_swin_add import DDIMDepthEstimate_Swin_ADD
from .ddim_depth_estimate_res_swin_addHAHI import DDIMDepthEstimate_Swin_ADDHAHI
from .ddim_depth_estimate_res_swin_addHAHI_vis import DDIMDepthEstimate_Swin_ADDHAHIVis
from .ddim_depth_estimate_res_mpvit_HAHI import DDIMDepthEstimate_MPVIT_ADDHAHI
from .ddim_depth_estimate_res import DDIMDepthEstimate_Res
from .ddim_depth_estimate_res_vis import DDIMDepthEstimate_ResVis


def get(args):
    model_name = args.head_name
    module_name = 'model.' + model_name.lower()
    module = import_module(module_name)

    return getattr(module, model_name)
