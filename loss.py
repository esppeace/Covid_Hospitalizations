# loss.py
import torch
import torch.nn as nn

def averageWIS(quantile, pred, target, device, naive=False, average=True):
    losses = []
    quantile = torch.tensor(quantile).to(device)
    target = target.reshape(-1, 1).repeat(1,len(quantile))
    if naive:
        pred = pred.reshape(-1, 1).repeat(1, len(quantile))

    errors = pred - target
    a = quantile*(target - pred)
    b = (1-quantile)*(pred - target)
    c = 2*torch.mean(torch.max(a, b),dim=1)
    if average:
        return torch.mean(c)
    else:
        return c

def Quantile_loss(quantile, pred, target, device, smooth=False, alpha=0.2):
    quantiles = torch.tensor(quantile)
    quantile = torch.tensor(quantile).to(device)
    target = target.reshape(-1, 1).repeat(1,len(quantiles))

    errors = target - pred
    a = quantile*(errors)
    b = (quantile-1)*(errors)

    c = torch.mean(torch.max(a, b),dim=1)
    
    return torch.mean(c)

def Huber_loss(prediction, targets, beta=1.0, cummulate=True):
    criterion = torch.nn.SmoothL1Loss(beta=beta)

    # combined next week 7 days predictions
    if cummulate:
        new_targets = torch.sum(targets, dim=1)
    else:
        new_targets = targets
    loss = criterion(prediction, new_targets)
    return loss

