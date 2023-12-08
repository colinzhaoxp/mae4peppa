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

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples_masked, samples, target) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples_masked = samples_masked.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, weight_pred = model(samples_masked, samples)
            weight_loss = log_rmse(weight_pred, target)
            loss = loss + weight_loss

            abs_acc, rel_acc = acc_metric(weight_pred, target)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value, weight_loss=weight_loss.item(),
                             absulate_acc=abs_acc.item(), relative_acc=rel_acc.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_weight_loss', weight_loss.item(), epoch_1000x)
            log_writer.add_scalar('train_absolute_acc', abs_acc, epoch_1000x)
            log_writer.add_scalar('train_relative_acc', rel_acc, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def log_rmse(preds, lables):
    clipped_preds = torch.clamp(preds, 1, float('inf'))
    rmse = torch.mean((torch.log(clipped_preds) - torch.log(lables)) ** 2)
    return rmse


def acc_metric(preds, lables):
    abs_acc = torch.mean(torch.abs(preds - lables))
    rel_acc = 1 - torch.mean(torch.abs(preds - lables) / lables)
    return abs_acc, rel_acc


def evaluate(model, dataloader, device, args):
    model.eval()
    abs_accs = []
    rel_accs = []
    with torch.no_grad():
        for data_iter_step, (samples_masked, target) in enumerate(dataloader):
            samples_masked = samples_masked.to(device, non_blocking=True)
            # samples = samples.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            loss, weight_pred = model(samples_masked)
            abs_acc, rel_acc = acc_metric(weight_pred, target)
            abs_accs.append(abs_acc.item())
            rel_accs.append(rel_acc.item())

    abs_acc = sum(abs_accs) / len(abs_accs)
    rel_acc = sum(rel_accs) / len(rel_accs)

    return abs_acc, rel_acc