# model/gpt.py

import torch
from torch import nn
import torch.nn.functional as F

from model.transformer import TransformerBlock
from model.rmsnorm import RMSNorm

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.n_embed)
        self.output_layer = nn.Linear(config.n_embed, config.vocab_size, bias = False)

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, use_cache = False, past_kvs = None):
        # input original token ids
        # shape x: [batch, seq_len]
        batch_size, seq_len = x.size()
        if seq_len == 0:
            raise ValueError("Input sequence length must be greater than 0")

        past_len = 0 if past_kvs is None else past_kvs[0][0].size(-2)
        if past_len + seq_len > self.config.block_size:
            raise ValueError(f"Input sequence length {seq_len} with past length {past_len} exceeds the maximum block size {self.config.block_size}")

        # 1. embedding layer
        x = self.token_embedding(x)  # shape: [batch, seq_len, embed_dim]


        present_kvs = []
        # 2. transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            x, present_kv = block(x, use_cache, past_kvs)

            if use_cache:
                present_kvs.append(present_kv)

        # 3. final layer norm
        x = self.norm(x)  # shape: [batch, seq_len, embed_dim]

        # 4. output layer - logits for next token prediction
        x = self.output_layer(x)  # shape: [batch, seq_len, vocab_size]

        return x, present_kvs if use_cache else None

