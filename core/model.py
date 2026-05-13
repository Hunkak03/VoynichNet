"""
model.py — VoynichTransformer
GPT-style causal language model for manuscript sequence analysis.
Replaces the shallow GRU with a proper multi-layer Transformer:
  - Sinusoidal positional encoding
  - Multi-head causal self-attention with attention map export
  - Pre-LN residual blocks (more stable training than post-LN)
  - GELU feed-forward sublayers
  - Input/output weight tying (cuts params ~20%, regularises)
  - Xavier/normal init for stable loss at step 0
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    vocab_size:   int
    emb_dim:      int   = 256
    n_heads:      int   = 8
    n_layers:     int   = 6
    ff_dim:       int   = 1024
    max_seq_len:  int   = 256
    dropout:      float = 0.10

    def __post_init__(self):
        if self.emb_dim % self.n_heads != 0:
            raise ValueError(f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})")

    @classmethod
    def small(cls, vocab_size: int) -> "ModelConfig":
        """Fast config for experimentation / short corpora."""
        return cls(vocab_size=vocab_size, emb_dim=128, n_heads=4, n_layers=4, ff_dim=512)

    @classmethod
    def base(cls, vocab_size: int) -> "ModelConfig":
        """Balanced quality-speed trade-off."""
        return cls(vocab_size=vocab_size, emb_dim=256, n_heads=8, n_layers=6, ff_dim=1024)

    @classmethod
    def large(cls, vocab_size: int) -> "ModelConfig":
        """Maximum capacity — needs GPU."""
        return cls(vocab_size=vocab_size, emb_dim=512, n_heads=8, n_layers=8, ff_dim=2048)


# ─── Positional Encoding ──────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, emb_dim: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, emb_dim)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float) * (-math.log(10_000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


# ─── Attention ────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Efficient fused QKV projection with causal mask.
    Returns (output, averaged_attention_weights) so callers can visualise attention.
    """

    def __init__(self, emb_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = emb_dim // n_heads
        self.scale    = self.head_dim ** -0.5

        self.qkv       = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.out_proj  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                      # each (B, H, T, D)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(mask[..., :T, :T] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = (weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.out_proj(out)), weights.mean(dim=1)  # avg over heads


# ─── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-LN block: LayerNorm → sublayer → residual (more stable than post-LN)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.emb_dim)
        self.attn = CausalSelfAttention(cfg.emb_dim, cfg.n_heads, cfg.dropout)
        self.ln2  = nn.LayerNorm(cfg.emb_dim)
        self.ff   = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.emb_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_w = self.attn(self.ln1(x), mask)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_w


# ─── Main Model ───────────────────────────────────────────────────────────────

class VoynichTransformer(nn.Module):
    """
    Causal Transformer LM for Voynich/cipher-text sequence modelling.

    Key design choices
    ------------------
    • Weight tying  — token embedding and output projection share weights.
                      Cuts ~vocab_size × emb_dim parameters; acts as implicit
                      regulariser (Press & Wolf, 2017).
    • Pre-LN blocks — avoids gradient vanishing in deep stacks.
    • Causal mask   — registered buffer, auto-moves with .to(device).
    • Two forward paths
        forward()           → last-position logits  (inference / autoregressive)
        forward_sequence()  → all-position logits   (training, full-sequence CE)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=0)
        self.pos_enc   = SinusoidalPE(cfg.emb_dim, cfg.max_seq_len, cfg.dropout)
        self.blocks    = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f      = nn.LayerNorm(cfg.emb_dim)
        self.lm_head   = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        # Causal mask: lower-triangular, shape (1, 1, max_len, max_len)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask)

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    # ── Forward passes ────────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor):
        h = self.pos_enc(self.token_emb(x))
        attention_maps = []
        for block in self.blocks:
            h, attn = block(h, self.causal_mask)
            attention_maps.append(attn.detach())
        return self.ln_f(h), attention_maps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns (last-pos logits, attention_maps). Used for autoregressive generation."""
        h, attn_maps = self._encode(x)
        return self.lm_head(h[:, -1, :]), attn_maps

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Returns all-position logits (B, T, V). Used during training."""
        h, _ = self._encode(x)
        return self.lm_head(h)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> str:
        total   = self.count_parameters()
        emb_p   = self.token_emb.weight.numel()
        lines = [
            f"  Architecture : Transformer (Pre-LN)",
            f"  Layers       : {self.cfg.n_layers}",
            f"  Heads        : {self.cfg.n_heads}",
            f"  Emb dim      : {self.cfg.emb_dim}",
            f"  FF dim       : {self.cfg.ff_dim}",
            f"  Vocab size   : {self.cfg.vocab_size}",
            f"  Total params : {total:,}  ({total/1e6:.2f}M)",
            f"  Emb params   : {emb_p:,}  (tied — not double-counted)",
        ]
        return "\n".join(lines)
