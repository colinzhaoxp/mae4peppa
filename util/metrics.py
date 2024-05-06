import torch


def compute_mae_acc(preds, labels):
    preds = preds.reshape(-1)
    mae_acc = torch.mean(torch.abs(preds - labels)) * 100
    return mae_acc.item()


def compute_mse_acc(preds, labels):
    preds = preds.reshape(-1)
    mse_acc = torch.mean((preds - labels) ** 2)
    return mse_acc.item()


def compute_mse(preds, labels):
    preds = preds.reshape(-1)
    mse = torch.sum((preds - labels) ** 2)
    return mse.item()


def compute_mape_acc(preds, labels):
    preds = preds.reshape(-1)
    mape_acc = torch.mean(torch.abs(preds - labels) / labels) * 100
    return mape_acc.item()


def compute_acc(preds, labels):
    acc = torch.sum(preds.argmax(1) == labels) / len(labels) * 100
    return acc.item()