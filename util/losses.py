import torch


def compute_mae(preds, labels):
    loss = torch.mean(torch.abs(preds - labels))
    return loss


def compute_mse(preds, labels):
    loss = torch.mean((preds - labels) ** 2)
    return loss


def compute_log_rmse(preds, lables):
    clipped_preds = torch.clamp(preds, 1, float('inf'))
    rmse = torch.mean((torch.log(clipped_preds) - torch.log(lables)) ** 2)
    return rmse