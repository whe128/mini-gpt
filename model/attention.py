# model/attention.py

import torch
from torch import nn

from .rope import apply_rotary_pos_emb

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embed % config.n_head == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = config.n_head
        self.head_dim = config.n_embed // config.n_head

        self.q_proj = nn.Linear(config.n_embed, config.n_embed, bias = False)
        self.k_proj = nn.Linear(config.n_embed, config.n_embed, bias = False)
        self.v_proj = nn.Linear(config.n_embed, config.n_embed, bias = False)

        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)

        self.register_buffer("mask", mask)
    def forward(self, x, use_cache = False, past_kv = None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # shape x: [batch, seq_len, embed_dim]
        # calculate Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # seperate the heads
        # shape q, k, v: [batch, seq_len, num_heads, head_dim]
        # use the view to reshape the tensor
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # if have past_kv, concatenate the past_kv with the current k and v
        past_len = 0
        if past_kv is not None:
            # past_kv[0]: past_k, past_kv[1]: past_v
            # shape past_kv: ([batch, num_heads, past_seq_len, head_dim], [batch, num_heads, past_seq_len, head_dim])
            past_len = past_kv[0].size(2)

        # rotate the new q and k with the rotary positional embedding
        q, k = apply_rotary_pos_emb(q, k, start_pos = past_len)

        # concatenate the past_kv with the current k and v
        # after rotating q, k, because the past_kv is already rotated, we don't need to rotate it again
        if past_kv is not None:
            # shape [batch, num_heads, past_seq_len, head_dim] + [batch, num_heads, seq_len, head_dim] = [batch, num_heads, past_seq_len + seq_len, head_dim]
            k = torch.cat([past_kv[0], k], dim = 2)
            v = torch.cat([past_kv[1], v], dim = 2)

        # assert seq_len + past_len <= self.mask.size(-1), "sequence length exceeds the maximum block size"
        total_len = k.size(2)

        if total_len > self.mask.size(-1):
            raise ValueError(f"sequence length {total_len} exceeds the maximum block size {self.mask.size(-1)}")

        # get the mask with the correct size
        # k, use all the pask k
        # q, use all the current q
        # shape : [1, 1 ]
        # attention score q @ k^T, shape: [batch, num_heads, q_seq_len, k_seq_len]
        # q_seq_len = seq_len
        # k_seq_len = total_len = past_len + seq_len
        # shape mask: [1, 1, q_seq_len, total_len]
        causal_mask = self.mask[:, :, past_len:total_len, :total_len]

        # calculate the attention scores
        # shape attn_scores: [batch, num_heads, q_seq_len, total_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # apply the mask to the attention scores
        # shape attn_scores: [batch, num_heads, q_seq_len, total_len]
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # calculate the attention weights
        # shape attn_weights: [batch, num_heads, q_seq_len, total_len]
        attn_weights = torch.softmax(attn_scores, dim = -1)

        # calculate the attention output
        # shape attn_output: [batch, num_heads, seq_len, head_dim]
        attn = torch.matmul(attn_weights, v)

        # combine the heads
        # shape attn: [batch, seq_len, embed_dim]
        attn_output = attn.transpose(1,2).contiguous().view(batch_size, seq_len, x.size(-1))

        present_kv = (k, v) if use_cache else None

        return attn_output, present_kv
