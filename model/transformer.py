# model/transformer.py

from torch import nn
from model.attention import CausalSelfAttention
from model.mlp import SwiGLU
from model.rmsnorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = CausalSelfAttention(config)
        self.mlp = SwiGLU(config.n_embed)
        self.norm1 = RMSNorm(config.n_embed)
        self.norm2 = RMSNorm(config.n_embed)

    def forward(self, x, use_cache = False, past_kv = None):
        # input shape: [batch, seq_len, embed_dim]
        # output shape: [batch, seq_len, embed_dim]
        # 1. pre-norm
        x = self.norm1(x)

        # 2. calculate the attention output
        attn_out, present_kv = self.attn(x, use_cache, past_kv)

        # 3. residual connection and
        x = x + attn_out

        # 4. post-norm
        x = self.norm2(x)

        # 5. MLP
        x = x + self.mlp(x)

        # why there is no norm, because the pre norm will run in the next forward for the input

        return x, present_kv
