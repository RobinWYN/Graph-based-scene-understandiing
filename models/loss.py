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
    
    
class PreTrainLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        
    def anchor_loss(self, preds, gt):
        num_preds = preds.shape[1]
        anchors = gt[..., -1, :].unsqueeze(1).repeat(1, num_preds, 1)
        loss, _ = torch.min(torch.norm(preds - anchors, p=2, dim=-1), dim=-1) # without harming multi-modality
        return loss
    
    def VIF_los(self, preds, gt_vif):
        #TODO: implement this loss method
        loss = 0
        return loss
    
    def forward(self, preds, gt, vif):
        loss_a = self.anchor_loss(preds, gt)
        loss_v = self.VIF_loss(preds, vif)
        loss = loss_a + self.alpha * loss_v
        return loss
        
        