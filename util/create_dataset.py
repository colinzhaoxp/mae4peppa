
from .peppa import build_peppa_dataset
from .peppa2depth import build_peppa2depth_dataset

__factory = {
    'peppa': build_peppa_dataset,
    'peppa2depth': build_peppa2depth_dataset,
}

def create_dataset(name, *args):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset:", name)
    return __factory[name](*args)