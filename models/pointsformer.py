import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, **kwargs):
        super(TransformerBlock, self).__init__()

    def forward(self, x):
        return x

class TransformerDown(nn.Module):
    def __init__(self, **kwargs):
        super(TransformerDown, self).__init__()

    def forward(self, x):
        return x

class Pointsformer(nn.Module):
    def __init__(self, num_classes=40, use_normals=True,
                 blocks=[1, 2, 1, 1], embed_channel=48, k_neighbors=32,
                 heads=8, expansion=2, reducer=2, **kwargs):
        super(Pointsformer, self).__init__()
        self.num_classes = num_classes
        channel = 6 if use_normals else 3
        self.linear = nn.Linear(channel, embed_channel)
        transformer_blocks = []
        for block_num in blocks:
            for _ in range(block_num):
                transformer_blocks.append(TransformerBlock())
            transformer_blocks.append(TransformerDown())




    def forward(self, x):
        return None

if __name__ == '__main__':
    print("===> testing pointNet with use_normals")
    data = torch.rand(10, 6, 1024)
    model = Pointsformer(k=40, use_normals=True)
    out = model(data)
    # print(f"x shape is: {out['logits'].shape} | trans_feat shape is: {out['trans_feat'].shape}")

