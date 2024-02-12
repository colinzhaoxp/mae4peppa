import torch

from util.metrics import *
from .get_meters import get_meter

def evaluate(model, dataloader, device, args):
    model.eval()
    mae_accs = []
    mape_accs = []
    with torch.no_grad():
        for data_iter_step, (samples, depths, target) in enumerate(dataloader):
            samples = samples.to(device)
            depths = depths.to(device)
            target = target.to(device) / 100

            weight_pred = model(samples, depths)
            mae_acc = compute_mae_acc(weight_pred, target)
            mape_acc = compute_mape_acc(weight_pred, target)

            mae_accs.append(mae_acc)
            mape_accs.append(mape_acc)

    mae_acc = sum(mae_accs) / len(mae_accs)
    mape_acc = sum(mape_accs) / len(mape_accs)

    return mae_acc, mape_acc


def minist_evaluate(model, dataloader, device, args):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data_iter_step, (samples, target) in enumerate(dataloader):
            samples = samples.to(device)
            target = target.to(device)

            weight_pred = model(samples)

            correct += weight_pred.argmax(dim=1).eq(target).sum().item()

    return correct / len(dataloader.dataset)