
from datasets.peppa import build_peppa_dataset
from datasets.peppa2depth import build_peppa2depth_dataset
from datasets.peppa2depath_no_mask import build_peppa2depth_no_mask_dataset
from datasets.peppa2depthV2 import build_peppa2depthV2_dataset

__factory = {
    'peppa': build_peppa_dataset,
    'peppa2depth': build_peppa2depth_dataset,
    'peppa2depth_no_mask': build_peppa2depth_no_mask_dataset,
    'peppa2depthV2': build_peppa2depthV2_dataset,
}

def create_dataset(name, *args):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset:", name)
    return __factory[name](*args)