import torch.nn as nn


class L1(nn.Module):
    def __init__(self, weight=1.0):
        super(L1, self).__init__()
        self.weight = weight
        self.criterion = nn.L1Loss()

    def forward(self, pred, gt):
        loss = self.criterion(pred, gt)
        return self.weight * loss

