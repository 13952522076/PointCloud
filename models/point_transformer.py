import torch
import torch.nn as nn
from point_tansformer_utils import index_points, square_distance, PointNetSetAbstraction
import torch.nn.functional as F
import numpy as np


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels) -> None:
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class PointTransformer(nn.Module):
    def __init__(self, num_classes=40, use_normals=True, num_points=1024,
                 num_blocks=4, num_beighbors=16, transformer_dim=512, **kwargs) -> None:
        super().__init__()
        channel = 6 if use_normals else 3
        self.fc1 = nn.Sequential(
            nn.Linear(channel, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, num_beighbors)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(num_blocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(num_points // 4 ** (i + 1), num_beighbors, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, transformer_dim, num_beighbors))

        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** num_blocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.num_blocks = num_blocks

    def forward(self, x):
        x = x.transpose(2,1)
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]
        for i in range(self.num_blocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
        res = self.fc2(points.mean(1))
        return {
            "logits": res
        }


if __name__ == '__main__':
    print("===> testing point transformer with use_normals")
    data = torch.rand(10, 6, 1024)
    model = PointTransformer()
    out = model(data)
    print(out["logits"].shape)
