import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class PredLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reg_loss = nn.SmoothL1Loss()

    def forward(self, pred, cls_pred, gt):

        dist = []
        for mod_idx in range(6):
            dist.append(
                torch.sqrt(
                    ((pred[:, mod_idx, -1] - gt[:, -1]) ** 2).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)  #[batch, 6]
        cls_gt = F.softmax(-dist, dim=1)
        _, min_idcs = dist.min(dim=1)
        row_idcs = torch.arange(len(min_idcs)).long()

        cls_loss = torch.sum(-F.log_softmax(cls_pred.squeeze(-1), dim=-1) * cls_gt.detach(), dim=-1).mean()

        pred = pred[row_idcs, min_idcs]
        reg_loss = self.reg_loss(pred, gt)

        loss = cls_loss + reg_loss

        return loss

class ClsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        loss = self.cls_loss(pred, gt)
        return loss