import torch
import torch.nn as nn

mae_loss = nn.L1Loss()
mse_loss = nn.MSELoss()
crossentropy_loss = nn.CrossEntropyLoss()

def compute_mae(preds, labels):
    preds = preds.reshape(-1)
    # loss = torch.sum(torch.abs(preds - labels))
    loss = mae_loss(preds, labels)
    return loss


def compute_mse(preds, labels):
    preds = preds.reshape(-1)
    labels = labels.float()
    loss = mse_loss(preds, labels)
    return loss


def compute_log_rmse(preds, lables):
    preds = preds.reshape(-1)
    lables = lables.float()
    clipped_preds = torch.clamp(preds, 1, float('inf'))
    rmse = torch.mean((torch.log(clipped_preds) - torch.log(lables)) ** 2)
    return rmse


def compute_cross_entropy(preds, labels):
    # preds = preds.to(torch.float64)
    loss = crossentropy_loss(preds, labels)
    return loss