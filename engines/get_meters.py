from util.metrics import *

def get_meter(preds, labels, meter_names):
    ret = dict()

    if 'mae' in meter_names:
        ret['mae'] = compute_mae_acc(preds, labels)

    if 'mse' in meter_names:
        ret['mse'] = compute_mae_acc(preds, labels)

    if 'log_rmse' in meter_names:
        ret['log_rmse'] = compute_mae_acc(preds, labels)

    if 'acc' in meter_names:
        ret['acc'] = compute_acc(preds, labels)

    return ret
