import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PointNetLoss', '']


class PointNetLoss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(PointNetLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.ce = nn.CrossEntropyLoss()
        self.register_buffer("I", torch.eye(64)[None, :, :])

    def feature_transform_reguliarzer(self, trans):
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - self.I, dim=(1, 2)))
        return loss

    def forward(self, net_out, targets):
        logits = net_out["logits"]
        trans_feat = net_out["trans_feat"]
        mat_diff_loss = self.feature_transform_reguliarzer(trans_feat)

        total_loss = self.ce(logits, targets) + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, net_out, targets):
        logits = net_out["logits"]
        return self.ce(logits, targets)
