"""tests/test_model.py – Tests for GPT model forward pass."""

import pytest
import torch
from src.model import GPT, CausalSelfAttention, FeedForward, TransformerBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
CONTEXT_LENGTH = 32
LAYERS = 2
HIDDEN_SIZE = 64
ATTENTION_HEADS = 4
DROPOUT = 0.0  # Disable dropout for deterministic tests


@pytest.fixture
def small_gpt():
    """Return a small GPT model suitable for unit tests."""
    return GPT(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        layers=LAYERS,
        hidden_size=HIDDEN_SIZE,
        attention_heads=ATTENTION_HEADS,
        dropout=DROPOUT,
    )


# ---------------------------------------------------------------------------
# Model forward pass
# ---------------------------------------------------------------------------

class TestGPTForward:
    def test_output_shape(self, small_gpt):
        """Logits should have shape (batch, seq_len, vocab_size)."""
        batch, seq_len = 2, 16
        input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
        logits = small_gpt(input_ids)
        assert logits.shape == (batch, seq_len, VOCAB_SIZE), (
            f"Expected ({batch}, {seq_len}, {VOCAB_SIZE}), got {logits.shape}"
        )

    def test_batch_size_one(self, small_gpt):
        """Model should handle batch size of 1."""
        input_ids = torch.randint(0, VOCAB_SIZE, (1, CONTEXT_LENGTH))
        logits = small_gpt(input_ids)
        assert logits.shape == (1, CONTEXT_LENGTH, VOCAB_SIZE)

    def test_sequence_length_one(self, small_gpt):
        """Model should handle a single-token sequence."""
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 1))
        logits = small_gpt(input_ids)
        assert logits.shape == (1, 1, VOCAB_SIZE)

    def test_full_context_length(self, small_gpt):
        """Model should handle the maximum context length without error."""
        input_ids = torch.randint(0, VOCAB_SIZE, (1, CONTEXT_LENGTH))
        logits = small_gpt(input_ids)
        assert logits.shape == (1, CONTEXT_LENGTH, VOCAB_SIZE)

    def test_output_is_finite(self, small_gpt):
        """All logits should be finite (no NaN or Inf)."""
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
        logits = small_gpt(input_ids)
        assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values."

    def test_causal_masking(self, small_gpt):
        """Changing future tokens should NOT affect past logits (causal property)."""
        small_gpt.eval()
        seq_len = 8
        input_ids = torch.randint(0, VOCAB_SIZE, (1, seq_len))

        logits_original = small_gpt(input_ids)

        # Modify all tokens after position 3
        modified_ids = input_ids.clone()
        modified_ids[:, 4:] = torch.randint(0, VOCAB_SIZE, (1, seq_len - 4))

        logits_modified = small_gpt(modified_ids)

        # Positions 0..3 should be unaffected
        assert torch.allclose(
            logits_original[:, :4, :], logits_modified[:, :4, :], atol=1e-5
        ), "Causal mask violation: modifying future tokens changed past logits."

    def test_num_parameters(self, small_gpt):
        """num_parameters should return a positive integer."""
        n = small_gpt.num_parameters()
        assert isinstance(n, int)
        assert n > 0

    def test_weight_tying(self, small_gpt):
        """Token embedding and LM head should share the same weight tensor."""
        assert small_gpt.token_embedding.weight is small_gpt.lm_head.weight


# ---------------------------------------------------------------------------
# Sub-module shapes
# ---------------------------------------------------------------------------

class TestCausalSelfAttention:
    def test_output_shape(self):
        attn = CausalSelfAttention(hidden_size=HIDDEN_SIZE, attention_heads=ATTENTION_HEADS, dropout=0.0)
        x = torch.randn(2, 10, HIDDEN_SIZE)
        out = attn(x)
        assert out.shape == x.shape

    def test_invalid_heads_raises(self):
        with pytest.raises(AssertionError):
            CausalSelfAttention(hidden_size=65, attention_heads=4, dropout=0.0)


class TestFeedForward:
    def test_output_shape(self):
        ffn = FeedForward(hidden_size=HIDDEN_SIZE, dropout=0.0)
        x = torch.randn(2, 10, HIDDEN_SIZE)
        out = ffn(x)
        assert out.shape == x.shape


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(hidden_size=HIDDEN_SIZE, attention_heads=ATTENTION_HEADS, dropout=0.0)
        x = torch.randn(2, 10, HIDDEN_SIZE)
        out = block(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# from_config helper
# ---------------------------------------------------------------------------

class TestFromConfig:
    def test_from_config(self):
        """GPT.from_config should produce the same model as direct instantiation."""
        from src.config import ModelConfig

        cfg = ModelConfig(
            vocab_size=VOCAB_SIZE,
            context_length=CONTEXT_LENGTH,
            layers=LAYERS,
            hidden_size=HIDDEN_SIZE,
            attention_heads=ATTENTION_HEADS,
            dropout=DROPOUT,
        )
        model = GPT.from_config(cfg)
        assert isinstance(model, GPT)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
        logits = model(input_ids)
        assert logits.shape == (1, 8, VOCAB_SIZE)
