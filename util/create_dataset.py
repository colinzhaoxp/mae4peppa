
from datasets.peppa import build_peppa_dataset
from datasets.peppa2depth import build_peppa2depth_dataset
from datasets.peppa2depath_no_mask import build_peppa2depth_no_mask_dataset
from datasets.peppa2depthV2 import build_peppa2depthV2_dataset
from datasets.cifar10test import build_cifar_dataset
from datasets.peppa2depthV3 import build_peppa2depthV3_dataset
from datasets.peppa2depthV3_mask import build_peppa2depthV3_mask_dataset
from datasets.peppa2depthV4 import build_peppa2depthV4_dataset
from datasets.peppa2depathV4tiny import build_peppa2depthV4tiny_dataset

__factory = {
    'peppa': build_peppa_dataset,
    'peppa2depth': build_peppa2depth_dataset,
    'peppa2depth_no_mask': build_peppa2depth_no_mask_dataset,
    'peppa2depthV2': build_peppa2depthV2_dataset,
    'cifar10': build_cifar_dataset,
    'peppa2depthV3': build_peppa2depthV3_dataset,
    'peppa2depthV3_mask': build_peppa2depthV3_mask_dataset,
    'peppa2depthV4': build_peppa2depthV4_dataset,
    'peppa2depthV4tiny': build_peppa2depthV4tiny_dataset,
}


def create_dataset(name, *args):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset:", name)
    return __factory[name](*args)