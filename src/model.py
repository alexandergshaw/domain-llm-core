"""model.py – Decoder-only GPT-style transformer implemented from scratch in PyTorch.

Architecture overview
---------------------
  TokenEmbedding  ──┐
  PositionalEmbedding─┘──► Embedding Dropout
                               │
               ┌───────────────┘
               │   ╔═ TransformerBlock (× N layers) ═╗
               │   ║  LayerNorm                       ║
               │   ║  CausalSelfAttention              ║
               │   ║  Residual add                     ║
               │   ║  LayerNorm                        ║
               │   ║  FeedForward MLP                  ║
               │   ║  Residual add                     ║
               │   ╚══════════════════════════════════╝
               │               │
               └───────────────┘
                      LayerNorm
                      Linear (vocab projection)
                      LogSoftmax (optional; cross-entropy wants logits)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Causal self-attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention.

    Each head attends only to positions ≤ current position, which prevents
    the model from peeking at future tokens during training.
    """

    def __init__(self, hidden_size: int, attention_heads: int, dropout: float):
        super().__init__()
        assert hidden_size % attention_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by attention_heads ({attention_heads})"
        )

        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.head_dim = hidden_size // attention_heads

        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            (batch, seq_len, hidden_size)
        """
        B, T, C = x.shape

        # Compute Q, K, V in a single matmul then split
        qkv = self.qkv_proj(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(self.hidden_size, dim=-1)

        # Reshape to (B, heads, T, head_dim) for batched attention
        def _reshape(t: Tensor) -> Tensor:
            return t.view(B, T, self.attention_heads, self.head_dim).transpose(1, 2)

        q, k, v = _reshape(q), _reshape(k), _reshape(v)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, heads, T, T)

        # Build causal mask (upper triangle → -inf so softmax → 0)
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        return self.resid_dropout(self.out_proj(out))


# ---------------------------------------------------------------------------
# Position-wise feed-forward MLP
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation and dropout.

    The hidden dimension is 4× the model hidden size, following the original
    Transformer paper.
    """

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """A single decoder block: pre-norm self-attention + pre-norm FFN.

    Pre-LayerNorm (applied before each sub-layer) tends to train more stably
    than the original post-LayerNorm formulation.
    """

    def __init__(self, hidden_size: int, attention_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, attention_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Residual connections wrap each sub-layer
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """Decoder-only GPT-style language model.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    context_length : int
        Maximum sequence length the model can handle.
    layers : int
        Number of TransformerBlock layers.
    hidden_size : int
        Embedding / hidden dimension.
    attention_heads : int
        Number of attention heads (must divide hidden_size evenly).
    dropout : float
        Dropout probability applied throughout the model.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        layers: int,
        hidden_size: int,
        attention_heads: int,
        dropout: float,
    ):
        super().__init__()

        # --- Embeddings -------------------------------------------------------
        # Token embedding: maps token IDs → dense vectors
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        # Positional embedding: learned absolute positions 0 … context_length-1
        self.position_embedding = nn.Embedding(context_length, hidden_size)
        self.embed_dropout = nn.Dropout(dropout)

        # --- Transformer body -------------------------------------------------
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, attention_heads, dropout)
            for _ in range(layers)
        ])

        # --- Output head -------------------------------------------------------
        self.ln_final = nn.LayerNorm(hidden_size)
        # Project hidden states back to vocabulary logits
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying: share token-embedding weights with the output projection.
        # This reduces parameters and often improves perplexity.
        self.lm_head.weight = self.token_embedding.weight

        # Initialise weights
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise linear and embedding layers with a small normal distribution."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids: (batch, seq_len) long tensor of token IDs.
        Returns:
            logits: (batch, seq_len, vocab_size) unnormalised log-probabilities.
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Build position indices [0, 1, …, T-1] for the current sequence
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)

        # Combine token and positional embeddings
        tok_emb = self.token_embedding(input_ids)        # (B, T, C)
        pos_emb = self.position_embedding(positions)     # (1, T, C)
        x = self.embed_dropout(tok_emb + pos_emb)

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and vocab projection
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of (trainable) parameters."""
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)

    @classmethod
    def from_config(cls, cfg) -> "GPT":
        """Instantiate GPT from a ModelConfig dataclass."""
        return cls(
            vocab_size=cfg.vocab_size,
            context_length=cfg.context_length,
            layers=cfg.layers,
            hidden_size=cfg.hidden_size,
            attention_heads=cfg.attention_heads,
            dropout=cfg.dropout,
        )
