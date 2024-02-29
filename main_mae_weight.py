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

from engines.engine_mae_weight import train_one_epoch
from engines.engine_test import evaluate_mae_weight, minist_evaluate
from util.create_dataset import create_dataset
from util.iotools import save_train_configs
from util.mylogging import Logger
from util.options import get_args_parser
from util.logger import setup_logger
from util.my_utils import save_checkpoint, load_checkpoint


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
        transforms.ToTensor()
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

    logger = setup_logger('MAE', save_dir=args.output_dir, if_train=True)

    sys.stdout = Logger(os.path.join(args.output_dir, 'log.txt'))
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

    # load pre-trained model
    if args.resume:
        model = load_checkpoint(model, args.resume)
    else:
        model = load_checkpoint(model, "/home/zhaoxp/workspace/mae-test/output_dir/mae_finetuned_vit_base.pth")

    model.to(device)

    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    logger.info(optimizer)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.perf_counter()
    val_best_mae_acc = float('inf')
    val_best_mae_epoch = 0

    for epoch in range(1, args.epochs+1):

        train_one_epoch(model, data_loader_train, optimizer, device, epoch,
                        log_writer=log_writer, args=args, logger=logger)

        if epoch % args.evaluate_period == 0:
            val_mae_acc, val_mape_acc = evaluate_mae_weight(model, data_loader_test, device, args)
            # val_mape_acc = 0
            # val_mae_acc = minist_evaluate(model, data_loader_test, device, args)
            info_str = f'Evaluate: mae_acc = {val_mae_acc:.4f}, mape_acc = {val_mape_acc:.4f}'

            if val_mae_acc < val_best_mae_acc:
                val_best_mae_acc = val_mae_acc
                val_best_mae_epoch = epoch

                save_checkpoint(model.state_dict(), args.output_dir + '/checkpoint-best_MAE_ACC.pth')

            log_stats = {'val_best_mae_epoch': val_best_mae_epoch, 'val_best_mae_acc': val_best_mae_acc,}

            for k, v in log_stats.items():
                info_str += f", {k}: {v:.4f} "

            logger.info(info_str)

    print("start evaluating on test dataset")
    model = load_checkpoint(model, args.output_dir + '/checkpoint-best_MAE_ACC.pth')
    mae_acc, mape_acc = evaluate_mae_weight(model, data_loader_test, device, args)
    print("Evaluate: mae_acc = %.4f, mape_acc = %.4f" % (mae_acc, mape_acc))

    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_train_configs(args.output_dir, args)
    main(args)
