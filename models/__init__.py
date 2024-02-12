from .models_mae import *
from .models_vit import *
from .models_weight import *
from .models_resnet import *
from .vits import *
from .convnext import *

from util.my_utils import load_checkpoint

__factory = {
    'mae_vit_base_patch16': mae_vit_base_patch16,
    'mae_vit_large_patch16': mae_vit_large_patch16,
    'mae_vit_huge_patch14': mae_vit_huge_patch14,
    'vit_base_patch16': vit_base_patch16,
    'vit_large_patch16': vit_large_patch16,
    'vit_huge_patch14': vit_huge_patch14,
    'mae_weight_vit_base_patch16': mae_weight_vit_base_patch16,
    'mae_weight_vit_large_patch16': mae_weight_vit_large_patch16,
    'mae_weight_vit_huge_patch14': mae_weight_vit_huge_patch14,
    'mae_weight_vit_tiny': mae_weight_vit_tiny,
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'vit_tiny': vit_tiny,
    'vit_base': vit_base,
    'convnext_tiny': convnext_tiny,
    'convnext_small': convnext_small,
    'convnext_base': convnext_base,
    'convnext_large': convnext_large,
}


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    model = __factory[name](*args, **kwargs)
    if not args.resume:
        model = load_checkpoint(model, "/home/zhaoxp/workspace/mae-test/output_dir/mae_pretrain_vit_base.pth")
    return model
