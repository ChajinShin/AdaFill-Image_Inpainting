import torch
import torch.nn as nn
import torch.nn.functional as F
from Tools.Utils import VGG19


class L1(nn.Module):
    def __init__(self, weight=1.0):
        super(L1, self).__init__()
        self.weight = weight
        self.criterion = nn.L1Loss()

    def forward(self, pred, gt):
        loss = self.criterion(pred, gt)
        return self.weight * loss


class NSGAN(nn.Module):
    def __init__(self, weight=1.0, is_disc=True):
        super(NSGAN, self).__init__()
        self.weight = weight
        self.is_disc = is_disc
        self.criterion = nn.BCELoss()

    def forward(self, fake, real=None):
        if self.is_disc:
            fake_label = torch.zeros_like(fake)
            real_label = torch.ones_like(real)
        else:
            real_label = torch.ones_like(fake)

        if self.is_disc:
            fake_loss = self.criterion(fake, fake_label)
            real_loss = self.criterion(real, real_label)
            return self.weight * (fake_loss + real_loss)
        else:  # generator
            real_loss = self.criterion(fake, real_label)
            return self.weight * real_loss


class PerceptualLoss(nn.Module):
    def __init__(self, device, weight=1.0):
        super(PerceptualLoss, self).__init__()
        self.weight = weight
        self.vgg = VGG19().to(device)
        self.L1loss = nn.L1Loss()

    def forward(self, pred, GT):
        GT_vgg = self.vgg(GT)
        pred_vgg = self.vgg(pred)

        loss = 0.0
        loss += self.L1loss(pred_vgg['relu1_1'], GT_vgg['relu1_1'])
        loss += self.L1loss(pred_vgg['relu2_1'], GT_vgg['relu2_1'])
        loss += self.L1loss(pred_vgg['relu3_1'], GT_vgg['relu3_1'])
        loss += self.L1loss(pred_vgg['relu4_1'], GT_vgg['relu4_1'])
        loss += self.L1loss(pred_vgg['relu5_1'], GT_vgg['relu5_1'])
        return self.weight * loss


class StyleLoss(nn.Module):
    def __init__(self, device, weight=1.0):
        super(StyleLoss, self).__init__()
        self.weight = weight
        self.vgg = VGG19().to(device)
        self.L1loss = nn.L1Loss()

    def forward(self, pred, GT):
        GT_vgg = self.vgg(GT)
        pred_vgg = self.vgg(pred)

        loss = 0.0
        loss += self.L1loss(self.gram_matrix(pred_vgg['relu2_2']), self.gram_matrix(GT_vgg['relu2_2']))
        loss += self.L1loss(self.gram_matrix(pred_vgg['relu3_4']), self.gram_matrix(GT_vgg['relu3_4']))
        loss += self.L1loss(self.gram_matrix(pred_vgg['relu4_4']), self.gram_matrix(GT_vgg['relu4_4']))
        loss += self.L1loss(self.gram_matrix(pred_vgg['relu5_2']), self.gram_matrix(GT_vgg['relu5_2']))
        return self.weight * loss

    def gram_matrix(self, X):
        N, C, H, W = X.size()

        f = X.view(N, C, H * W)
        f_T = f.transpose(1, 2)
        # bmm -> batch matrix multiplication
        gram = torch.bmm(f, f_T) / (C * H * W)   # (N, C, C)

        return gram





