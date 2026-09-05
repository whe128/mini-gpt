# model/rope.py

import torch

def rotate_half(x):
    """
        rotate the last dimension of the input tensor by half
        every two elements in the last dimension are rotated
        rotation: multiply with the rotation matrix

        here: rotate 90 degrees
    """
    # shape x: [batch, num_heads, seq_len, head_dim]
    # ... means the previous dimensions keep
    #
    # shape x1: [batch, num_heads,seq_len,  head_dim // 2]
    # shape x2: [batch, num_heads, seq_len, head_dim // 2]
    x1 = x[..., ::2]  # even indices
    x2 = x[..., 1::2] # odd indices

    # rotate 90 degrees clockwise
    # [x1, x2] -> [-x2, x1]
    # first need stack x1 and x2 along the last dimension
    # shape [batch, num_heads, seq_len, head_dim // 2, 2]
    stacked = torch.stack([-x2, x1], dim=-1)

    # flatten the last two dimensions to get back to the original shape
    # shape [batch, num_heads, seq_len, head_dim]
    # if flatten -1, the tensor will not change the shape
    flattened = stacked.flatten(start_dim = -2)

    return flattened

def apply_rotary_pos_emb(q, k, start_pos):
    """
        apply the rotate_half function to the query and key tensors
        q: query tensor, shape [batch, seq_len, num_heads, head_dim]
        k: key tensor, shape [batch, seq_len, num_heads, head_dim]

        start_pos: the starting pos in the whole sequence for this token
    """
    # shape q: [batch, seq_len, num_heads, head_dim]
    # shape k: [batch, seq_len, num_heads, head_dim]

    seq_len = q.shape[-2]
    head_dim = q.shape[-1]
    device = q.device

    # compute the inverse frequency for each dimension in the head_dim
    # let each dimension in the head_dim corresponds to a different frequency
    # shape invers_freqs: [head_dim // 2]
    invers_freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device = device).float()/head_dim))

    # shape pos_seq: [seq_len]
    pos_seq = torch.arange(start_pos, start_pos + seq_len, device = device).float()

    # outter product of pos_seq and invers_freqs to get the rotation angles
    # pos_seq shape: [seq_len]
    # invers_freqs shape: [head_dim // 2]

    #       dim0   dim1   dim2   dim3
    # pos0  0*rot0  0*rot1  0*rot2  0*rot3
    # pos1  1*rot0  1*rot1  1*rot2  1*rot3
    # pos2  2*rot0  2*rot1  2*rot2  2*rot3
    # pos3  3*rot0  3*rot1  3*rot2  3*rot3

    # shape freqs: [seq_len, head_dim // 2]
    # element inside is the rotation angle for ith pos and jth dim
    freqs = torch.outer(pos_seq, invers_freqs)


    # recover the freqs to the original shape of q and k
    # shape freqs: [seq_len, head_dim]
    # emb = torch.stack((freqs, freqs), dim = -1).flatten(-2)
    emb = torch.cat((freqs, freqs), dim = -1)

    # None is used to add a new dimension to the tensor
    # calculate the cos and sin of the rotation angles
    # shape cos: [1, seq_len, 1, head_dim]
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]

    # rotateion matrix
    # R(θ) = [[cos(θ), -sin(θ)],
    #         [sin(θ), cos(θ)]]
    # x1 = x1 * cos - x2 * sin
    # x2 = x1 * sin + x2 * cos

    # q1 = q1 * cos - q2 * sin
    # q2 = q1 * sin + q2 * cos

    # rotate_half(q): q1 = -q2, q2 = q1
    # q1 = q1 * cos - q2 * sin
    # q2 = q2 * cos + q1 * sin
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin

    return q, k
