import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch
from torch import einsum
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from typing import Any




def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids



def index_points2(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

## validate the index function
# x = torch.rand(4,20,3)
# point=5
# indexes = farthest_point_sample(x[:, :, :3], point)
# sampeld1 = index_points(x, indexes)
# sampeld2 = index_points2(x, indexes)
# print(sampeld1-sampeld2)


# validate the knn function
distances = torch.rand(1,5,10)
k =5
knn_idx1 = distances.argsort()[:, :, :k]
_, knn_idx2 = torch.topk(distances, k, dim = -1, largest=False, sorted=False)
print(knn_idx1)
print(knn_idx2)
