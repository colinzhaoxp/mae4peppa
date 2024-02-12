# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import logging
import torch

import util.misc as misc
import util.lr_sched as lr_sched
from util.meter import AverageMeter
from engines.get_losses import get_loss
from engines.get_meters import get_meter


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    log_writer=None,
                    args=None,
                    logger=None
                    ):
    model.train()

    meters = {
        'loss': AverageMeter()
    }

    accum_iter = args.accum_iter

    for data_iter_step, (samples, depths, targets) in enumerate(data_loader):

        # we use a per iteration (instead of per epoch) lr scheduler
        # if data_iter_step % accum_iter == 0:
        #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device)
        depths = depths.to(device)
        targets = targets.to(device) / 100

        weight_pred = model(samples, depths)

        loss_ret = get_loss(weight_pred, targets, args.loss_names)
        meter_ret = get_meter(weight_pred, targets, args.meter_names)

        total_loss = sum([v for k, v in loss_ret.items() if 'loss' in k])

        # loss 输出
        meters['loss'].update(total_loss.item(), samples.size(0))
        for key in loss_ret.keys():
            if key not in meters.keys():
                meters.update({key: AverageMeter()})
            meters[key].update(loss_ret[key].item())

        # meter 输出
        for key in meter_ret.keys():
            if key not in meters.keys():
                meters.update({key: AverageMeter()})
            meters[key].update(meter_ret[key])

        if (data_iter_step + 1) % args.log_period == 0:
            info_str = f'Epoch[{epoch}] Iteration[{data_iter_step+1}/{len(data_loader)}]'
            for k, v in meters.items():
                info_str += f", {k}: {v.val:.4f}({v.avg:.4f})"

            info_str += f", lr: {optimizer.param_groups[0]['lr']:.6f}"

            logger.info(info_str)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            for k, v in meters.items():
                log_writer.add_scalar(k, v.val, data_iter_step + epoch * len(data_loader))

    #     break
    # model.eval()
    # with torch.no_grad():
    #     weight_pred2 = model(samples)
    #     print(weight_pred2)