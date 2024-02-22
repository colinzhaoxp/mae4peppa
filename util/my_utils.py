import os
import errno
import os.path as osp
import shutil

import torch
from torch.nn import Parameter
import yaml
from easydict import EasyDict as edict

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        # checkpoint = torch.load(fpath)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


def load_checkpoint(model, finetune_path):
    checkpoint = torch.load(finetune_path, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % finetune_path)
    if 'model' in checkpoint.keys():
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint

    model = copy_state_dict(checkpoint_model, model)

    return model

def load_train_configs(path):
    with open(path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return edict(args)