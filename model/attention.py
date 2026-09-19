"""Causal multi-head self-attention, written out by hand."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must divide evenly into n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # One matmul produces Q, K and V together, then we split it.
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Lower-triangular mask: position t may only look at <= t.
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim) so heads attend independently.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product: how much each position attends to every other.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v                                    # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-merge the heads
        return self.resid_dropout(self.proj(y))


def _self_check():
    """A future token must never influence an earlier one."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(n_embd=32, n_head=4, block_size=8, dropout=0.0).eval()
    x = torch.randn(1, 8, 32)
    out_a = attn(x)

    x2 = x.clone()
    x2[:, 5:, :] = torch.randn(1, 3, 32)  # scramble the future
    out_b = attn(x2)

    assert torch.allclose(out_a[:, :5], out_b[:, :5], atol=1e-6), "causal mask leaks future tokens"
    assert out_a.shape == x.shape
    print("attention ok: causal mask holds, shapes preserved")


if __name__ == "__main__":
    _self_check()
