import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BCELogLoss(nn.Module):
    def __init__(self, device, smooth=False):
        super(BCELogLoss, self).__init__()
        self.device = device
        # label smoothing
        self.smooth = smooth
    def forward(self, output, target):
        target = torch.abs(target)
        weights = list()
        for index in range(target.size()[1]):
            _target = target[:, index].view(-1)
            if _target.sum() == 0:
                weights.append(0.0)
            else:
                weight = (_target.size()[0] - _target.sum()) / _target.sum()
                weights.append(weight)
        # Label smoothing
        if self.smooth and np.random.choice([True, False], p=[0.3, 0.7]):
            eps = np.random.uniform(0.05, 0.15)
            target = (1-eps)*target + eps / target.size()[1]

        weights = torch.FloatTensor(weights).to(self.device)
        return F.binary_cross_entropy_with_logits(output, target, pos_weight=weights)

class WeightedBCELoss(nn.Module):
    def __init__(self, device, weights=None):
        super(WeightedBCELoss, self).__init__()
        self.device = device
        self.weights = torch.FloatTensor(weights)
        self.weights = (self.weights/self.weights.sum()).to(self.device).unsqueeze(0)

    def forward(self, output, target):
        target = torch.abs(target)
        weights = list()
        for index in range(target.size()[1]):
            _target = target[:, index].view(-1)
            if _target.sum() == 0:
                weights.append(0.0)
            else:
                weight = (_target.size()[0] - _target.sum()) / _target.sum()
                weights.append(weight)

        weights = torch.FloatTensor(weights).to(self.device)
        loss = F.binary_cross_entropy_with_logits(output, target, pos_weight=weights, reduction='none')
        return (loss.mean(0)*self.weights).sum()