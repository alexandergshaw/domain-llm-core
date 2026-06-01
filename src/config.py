"""config.py – Load YAML configuration into a typed dataclass."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    # Vocabulary / tokenizer
    vocab_size: int = 8000

    # Sequence
    context_length: int = 512

    # Transformer architecture
    layers: int = 6
    hidden_size: int = 384
    attention_heads: int = 6
    dropout: float = 0.1

    # Training hyper-parameters
    batch_size: int = 16
    learning_rate: float = 3e-4
    epochs: int = 5

    # Optional fields that may be added by future config files
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        """Load a YAML file and return a ModelConfig instance."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open() as fh:
            data = yaml.safe_load(fh) or {}

        known_fields = {f for f in cls.__dataclass_fields__ if f != "extra"}
        init_kwargs: dict = {}
        extra: dict = {}

        for key, value in data.items():
            if key in known_fields:
                init_kwargs[key] = value
            else:
                extra[key] = value

        init_kwargs["extra"] = extra
        return cls(**init_kwargs)

    def __repr__(self) -> str:
        lines = ["ModelConfig("]
        for key, val in self.__dict__.items():
            if key != "extra" or val:
                lines.append(f"  {key}={val!r},")
        lines.append(")")
        return "\n".join(lines)
