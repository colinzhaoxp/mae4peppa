# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import numpy as np
import os
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models

from engines.engine_weight_train import train_one_epoch
from engines.engine_test import evaluate, minist_evaluate
from util.create_dataset import create_dataset
from util.iotools import save_train_configs
from util.mylogging import Logger
from util.options import get_args_parser
from util.logger import setup_logger
from util.my_utils import save_checkpoint


def load_checkpoint(model, finetune_path):
    checkpoint = torch.load(finetune_path, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % finetune_path)
    checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    return model


def get_dataset(args):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if args.in_chans == 4:
        depth_norm = (0.005261, 0.011198)
        mean.append(depth_norm[0])
        std.append(depth_norm[1])

    # simple augmentation
    transform_train = transforms.Compose([
        transforms.Resize(args.input_size, interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=(0, 360)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset_train, dataset_val, dataset_test = create_dataset(args.data_name, args, transform_train)

    return dataset_train, dataset_val, dataset_test


def get_dataloader(args, dataset_train, dataset_val, dataset_test):
    # sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )

    return data_loader_train, data_loader_val, data_loader_test


def main(args):

    if args.log_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    logger = setup_logger('MAE', save_dir=args.output_dir, if_train=False)

    sys.stdout = Logger(os.path.join(args.output_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, dataset_val, dataset_test = get_dataset(args)

    if dataset_test is None:
        dataset_test = dataset_val

    data_loader_train, data_loader_val, data_loader_test = get_dataloader(args, dataset_train, dataset_val, dataset_test)

    # define the model
    model = models.create(name=args.model, img_size=args.input_size, in_chans=args.in_chans, class_num=args.class_num,
                          patch_size=args.patch_size)
    model.to(device)

    print("start evaluating on test dataset")
    model = load_checkpoint(model, args.resume)
    mae_acc, mape_acc = evaluate(model, data_loader_test, device, args)
    print("Evaluate: mae_acc = %.4f, mape_acc = %.4f" % (mae_acc, mape_acc))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.resume:
        args.output_dir = os.path.dirname(args.resume)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
