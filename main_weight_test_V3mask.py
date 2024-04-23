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

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models

from engines.engine_weight_train import train_one_epoch
from engines.engine_test import evaluate_mae_weight
from util.create_dataset import create_dataset
from util.iotools import save_train_configs
from util.mylogging import Logger
from util.options import get_args_parser
from util.logger import setup_logger
from util.my_utils import save_checkpoint, load_checkpoint, load_train_configs


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
        transforms.RandomRotation(degrees=(0, 360)),
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


def change_root_file_path(data, new_root_path):
    rgb_root_path = os.path.join(new_root_path, "test_mask_recon")
    depth_root_path = os.path.join(new_root_path, "test_mask_recon_dep")
    for i in range(len(data)):
        origin_rgb_path, origin_depth_path, weight, mask_rgb_path, mask_depth_path = data[i]
        mask_rgb_path = os.path.join(rgb_root_path, os.path.basename(mask_rgb_path))
        mask_depth_path = os.path.join(depth_root_path, os.path.basename(mask_depth_path))
        data[i] = (origin_rgb_path, origin_depth_path, weight, mask_rgb_path, mask_depth_path)


def main(args):

    if args.log_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(args.output_dir, 'log_test-V3-mask.txt'))
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

    if args.new_data_path:
        change_root_file_path(dataset_test.dataset, args.new_data_path)

    data_loader_train, data_loader_val, data_loader_test = get_dataloader(args, dataset_train, dataset_val, dataset_test)

    # define the model
    model = models.create(name=args.model, img_size=args.input_size, in_chans=args.in_chans, class_num=args.class_num,
                          patch_size=args.patch_size)
    model.to(device)

    print("start evaluating on test dataset")
    model = load_checkpoint(model, os.path.join(args.output_dir, 'checkpoint-best_MAE_ACC.pth'))
    mae_acc, mape_acc = evaluate_mae_weight(model, data_loader_test, device, args)
    print("Evaluate: mae_acc = %.4f, mape_acc = %.4f" % (mae_acc, mape_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="weight model test for V3 mask dataset")
    parser.add_argument("--config_file", default='/home/zhaoxp/workspace/mae-test/output_dir/2-13/configs.yaml')
    parser.add_argument('--data_name', default='peppa2depthV3', type=str, help='dataset name')
    parser.add_argument('--data_path', default='./data/peppa2depthV3', type=str,
                        help='dataset path')
    parser.add_argument("--new_data_path", default="", type=str, help="new data path")
    args = parser.parse_args()

    config_args = load_train_configs(args.config_file)
    config_args.data_name = args.data_name
    config_args.data_path = args.data_path
    config_args.new_data_path = args.new_data_path

    main(config_args)