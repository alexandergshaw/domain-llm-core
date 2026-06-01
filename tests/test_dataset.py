"""tests/test_dataset.py – Tests for TokenDataset shape and behaviour."""

import pytest
import torch
from src.dataset import TokenDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONTEXT_LENGTH = 16


@pytest.fixture
def token_ids():
    """A simple increasing sequence of token IDs."""
    return list(range(200))  # 200 tokens → 184 samples at context_length=16


@pytest.fixture
def dataset(token_ids):
    return TokenDataset(token_ids, context_length=CONTEXT_LENGTH)


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------

class TestDatasetLength:
    def test_length(self, dataset, token_ids):
        """Dataset length should be len(token_ids) - context_length."""
        expected = len(token_ids) - CONTEXT_LENGTH
        assert len(dataset) == expected

    def test_length_minimal(self):
        """Dataset with exactly context_length + 1 tokens should have 1 sample."""
        ids = list(range(CONTEXT_LENGTH + 1))
        ds = TokenDataset(ids, context_length=CONTEXT_LENGTH)
        assert len(ds) == 1

    def test_too_short_raises(self):
        """Providing fewer tokens than context_length + 1 should raise ValueError."""
        with pytest.raises(ValueError):
            TokenDataset(list(range(CONTEXT_LENGTH)), context_length=CONTEXT_LENGTH)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

class TestDatasetShape:
    def test_item_shapes(self, dataset):
        """Each item should return (input_ids, target_ids) both of length context_length."""
        input_ids, target_ids = dataset[0]
        assert input_ids.shape == (CONTEXT_LENGTH,), f"input shape: {input_ids.shape}"
        assert target_ids.shape == (CONTEXT_LENGTH,), f"target shape: {target_ids.shape}"

    def test_item_dtype(self, dataset):
        """Token tensors should be long (int64)."""
        input_ids, target_ids = dataset[0]
        assert input_ids.dtype == torch.long
        assert target_ids.dtype == torch.long

    def test_all_items_same_shape(self, dataset):
        """All samples should have the same shape."""
        for i in (0, len(dataset) // 2, len(dataset) - 1):
            inp, tgt = dataset[i]
            assert inp.shape == (CONTEXT_LENGTH,)
            assert tgt.shape == (CONTEXT_LENGTH,)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestDatasetCorrectness:
    def test_target_is_shifted_input(self, dataset, token_ids):
        """target_ids should equal input_ids shifted one position to the right."""
        for i in (0, 1, len(dataset) - 1):
            input_ids, target_ids = dataset[i]
            # targets[j] == inputs[j+1] == token_ids[i + j + 1]
            expected_input = torch.tensor(token_ids[i : i + CONTEXT_LENGTH], dtype=torch.long)
            expected_target = torch.tensor(token_ids[i + 1 : i + CONTEXT_LENGTH + 1], dtype=torch.long)
            assert torch.equal(input_ids, expected_input), f"input mismatch at idx={i}"
            assert torch.equal(target_ids, expected_target), f"target mismatch at idx={i}"

    def test_last_index(self, dataset, token_ids):
        """Last valid index should not raise."""
        last_idx = len(dataset) - 1
        input_ids, target_ids = dataset[last_idx]
        assert input_ids.shape == (CONTEXT_LENGTH,)

    def test_dataloader_batch(self, dataset):
        """The dataset should work with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch_input, batch_target = next(iter(loader))
        assert batch_input.shape == (4, CONTEXT_LENGTH)
        assert batch_target.shape == (4, CONTEXT_LENGTH)
