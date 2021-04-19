import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np

__all__ = ["MLP_max", "MLP_avg"]

class MLP(nn.Module):
    def __init__(self, num_classes=40, use_normals=True, pool='max', **kwargs):
        super(MLP, self).__init__()
        if use_normals:
            channel = 6
        else:
            channel = 3
        self.feat = nn.Sequential(
            nn.Conv1d(channel, 64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024,1),
            nn.BatchNorm1d(1024)
        )
        self.pool = nn.AdaptiveMaxPool1d(1) if pool == "max" else nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)


    def forward(self, x):
        x = self.feat(x)
        x = self.pool(x).squeeze(dim=-1)
        x = F.relu(self.bn1(self.fc1(x)), inplace=True)
        x = F.relu(self.bn2(self.dropout(self.fc2(x))), inplace=True)
        x = self.fc3(x)
        return {
            "logits":x
        }


def MLP_max(num_classes=40, **kwargs) -> MLP:
    return MLP(num_classes=num_classes, pool="max", **kwargs)

def MLP_avg(num_classes=40, **kwargs) -> MLP:
    return MLP(num_classes=num_classes, pool="avg", **kwargs)


if __name__ == '__main__':
    print("===> testing pointNet with use_normals")
    data = torch.rand(10, 6, 1024)
    model = MLP_avg(k=40, use_normals=True)
    out = model(data)
    print(f"x shape is: {out['logits'].shape}")
