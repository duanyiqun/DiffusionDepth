from importlib import import_module
from .mmbev_resnet import ResNetForMMBEV


def get(args):
    model_name = args.backbone_name
    module_name = args.backbone_module
    module_name = 'model.' + 'backbone.' + module_name.lower()
    module = import_module(module_name)

    return getattr(module, model_name)


