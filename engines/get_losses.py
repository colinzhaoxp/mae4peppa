import torch
from util.losses import *


def get_loss(preds, labels, loss_names):
    ret = dict()

    if 'mae' in loss_names:
        ret['mae_loss'] = compute_mae(preds, labels)

    if 'mse' in loss_names:
        ret['mse_loss'] = compute_mse(preds, labels)

    if 'log_rmse' in loss_names:
        ret['log_rmse_loss'] = compute_mse(preds, labels)

    return ret