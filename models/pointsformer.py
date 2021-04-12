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


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            # nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5  # the 1/sqrt(d_k) in Eq.1 in Attention all you need
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x):
        # x shape: [b, n, d] batch = batch*fps, n is neighbors, d is dimension.
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self,dim, **kwargs):
        super(TransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attention = Attention()
        self.ffn = FeedForward()

    def forward(self, x):
        # x should follow the shape of [b, p, k, d]
        b, p, k, d = x.shape  # batch, p fps (farthest points sampling), k neighbors, d dimnesion

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
    print("===> testing mode ...")
    data = torch.rand(10, 6, 128)
    model = Attention(128)
    out = model(data)
    print(out.shape)
    # print(f"x shape is: {out['logits'].shape} | trans_feat shape is: {out['trans_feat'].shape}")
