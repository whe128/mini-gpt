import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps = 1e-8):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        # scale_weight is a learnable parameter that scales the normalized output
        # initialized to 1
        self.scale_weight = nn.Parameter(torch.ones(embed_dim))
    def forward(self, x):
        # input shape: [batch, seq_len, embed_dim]
        # output shape: [batch, seq_len, embed_dim]

        # mean at the last dimension (embed_dim)
        # keep the dimension for the mean dimension with 1
        # shape rms: [batch, seq_len, 1]
        rms = x.pow(2).mean(-1, keepdim=True)

        return x * torch.rsqrt(rms + self.eps) * self.scale_weight

