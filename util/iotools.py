import os
import yaml
from easydict import EasyDict as edict

def save_train_configs(path, args, is_test=False):
    if not os.path.exists(path):
        os.makedirs(path)
    if is_test:
        fpath = f'{path}/test_configs.yaml'
    else:
        fpath = f'{path}/configs.yaml'
    with open(fpath, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

def load_train_configs(path):
    with open(path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return edict(args)