"""Decoder-only transformer (GPT) in raw PyTorch.

Attention, LayerNorm and the transformer block are written by hand.
Only nn.Linear, nn.Embedding, nn.Parameter and nn.Dropout are used as building blocks.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 128
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    # Ablation flags
    use_pos_emb: bool = True
    use_layernorm: bool = True
    use_residual: bool = True


class BigramModel(nn.Module):
    """Baseline: predicts the next token from the current token only, using a lookup table."""

    def __init__(self, cfg):
        super().__init__()
        self.table = nn.Embedding(cfg.vocab_size, cfg.vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)                                      # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class LayerNorm(nn.Module):
    """Normalise each vector over its last dimension to mean 0, variance 1,
    then apply a learnable scale (gamma) and shift (beta)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention with a causal mask."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_k = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)  # Q, K, V projections in one matrix
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Lower-triangular mask: position t may attend only to positions <= t.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                         # each (B, T, C)
        # Split C into n_head heads of size d_k: (B, T, C) -> (B, n_head, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)        # (B, n_head, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        weights = self.attn_drop(F.softmax(scores, dim=-1))
        out = weights @ v                                             # (B, n_head, T, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T, C)          # concatenate heads
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.fc2 = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln1(x)), then x + ffn(ln2(x))."""

    def __init__(self, cfg):
        super().__init__()
        norm = LayerNorm if cfg.use_layernorm else (lambda dim: nn.Identity())
        self.ln1 = norm(cfg.d_model)
        self.ln2 = norm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn = FeedForward(cfg)
        self.use_residual = cfg.use_residual

    def forward(self, x):
        if self.use_residual:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.attn(self.ln1(x))
            x = self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model) if cfg.use_pos_emb else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.d_model) if cfg.use_layernorm else nn.Identity()
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. Returns logits (B, T, vocab_size) and the loss if targets are given."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        x = self.tok_emb(idx)                                         # (B, T, C)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))  # (T, C) broadcast over B
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))                              # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively append `max_new_tokens` tokens to idx (B, T)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]                  # crop to context length
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature                   # last position only
            if top_k is not None:
                kth = torch.topk(logits, top_k).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
