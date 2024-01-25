from util.metrics import *

def evaluate(model, dataloader, device, args):
    model.eval()
    mae_accs = []
    mape_accs = []
    with torch.no_grad():
        for data_iter_step, (samples_masked, target) in enumerate(dataloader):
            samples_masked = samples_masked.to(device, non_blocking=True)
            # samples = samples.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            weight_pred = model(samples_masked)

            mae_acc = compute_mae_acc(weight_pred, target)
            mape_acc = compute_mape_acc(weight_pred, target)

            mae_accs.append(mae_acc)
            mape_accs.append(mape_acc)

    mae_acc = sum(mae_accs) / len(mae_accs)
    mape_acc = sum(mape_accs) / len(mape_accs)

    return mae_acc, mape_acc