"""dataset.py – PyTorch Dataset for next-token prediction.

Usage
-----
    from tokenizers import Tokenizer
    from src.dataset import TokenDataset

    tokenizer = Tokenizer.from_file("tokenizer.json")
    ids = tokenizer.encode(open("data/train.txt").read()).ids
    dataset = TokenDataset(ids, context_length=512)
    # dataset[i] → (input_ids, target_ids) both shape (context_length,)
"""

from __future__ import annotations

from typing import Sequence
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """A sliding-window dataset over a flat list of token IDs.

    Each sample is a pair ``(input_ids, target_ids)`` of length
    ``context_length`` where ``target_ids[i] = input_ids[i + 1]``.

    Parameters
    ----------
    token_ids : Sequence[int]
        A flat list (or array) of integer token IDs representing the
        entire training corpus.
    context_length : int
        The number of tokens in each input/target sequence.  The dataset
        produces ``len(token_ids) - context_length`` samples.
    """

    def __init__(self, token_ids: Sequence[int], context_length: int):
        if len(token_ids) <= context_length:
            raise ValueError(
                f"token_ids length ({len(token_ids)}) must be greater than "
                f"context_length ({context_length}) to form at least one sample."
            )
        # Store as a long tensor for zero-copy slicing
        self._ids: Tensor = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        # Each window of (context_length + 1) tokens yields one sample
        return len(self._ids) - self.context_length

    def __getitem__(self, idx: int):
        """Return ``(input_ids, target_ids)`` slices of length ``context_length``.

        Targets are inputs shifted by one position to the right, so the
        model learns to predict the next token at every position.
        """
        chunk = self._ids[idx : idx + self.context_length + 1]
        input_ids = chunk[:-1]   # tokens 0 … context_length-1
        target_ids = chunk[1:]   # tokens 1 … context_length
        return input_ids, target_ids
