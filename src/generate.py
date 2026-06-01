"""generate.py – Autoregressive text generation from a trained GPT checkpoint.

Example
-------
    python src/generate.py \\
        --checkpoint checkpoints/best.pt \\
        --tokenizer-path tokenizer.json \\
        --config configs/tiny.yaml \\
        --prompt "Once upon a time" \\
        --max-new-tokens 200 \\
        --temperature 0.8 \\
        --top-k 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero-out all logits except the top-k, then return the masked logits."""
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[:, -1].unsqueeze(-1)  # (batch, 1)
    return logits.masked_fill(logits < threshold, float("-inf"))


@torch.inference_mode()
def generate(
    model: "GPT",  # noqa: F821  (imported at runtime)
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """Autoregressively generate ``max_new_tokens`` tokens.

    Parameters
    ----------
    model:
        A trained GPT model in eval mode.
    input_ids:
        Seed token IDs, shape ``(1, prompt_len)``.
    max_new_tokens:
        Number of new tokens to generate.
    temperature:
        Sampling temperature. Values < 1.0 sharpen the distribution;
        values > 1.0 make it flatter. Use 1.0 for unmodified logits.
    top_k:
        If > 0, restrict sampling to the ``top_k`` most likely tokens.

    Returns
    -------
    torch.Tensor of shape ``(1, prompt_len + max_new_tokens)``.
    """
    context_length: int = model.position_embedding.num_embeddings

    for _ in range(max_new_tokens):
        # Trim the context window if it exceeds the model's maximum length
        ctx = input_ids[:, -context_length:]

        logits = model(ctx)               # (1, T, vocab_size)
        next_logits = logits[:, -1, :]    # only the last position

        # Apply temperature scaling
        if temperature != 1.0:
            next_logits = next_logits / temperature

        # Apply top-k filtering
        next_logits = top_k_logits(next_logits, top_k)

        # Sample from the (possibly filtered) distribution
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

        input_ids = torch.cat([input_ids, next_id], dim=1)

    return input_ids


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained domain-llm-core checkpoint."
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to the model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--tokenizer-path", required=True, type=Path,
        help="Path to the HuggingFace tokenizer.json file.",
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to the YAML model config (e.g. configs/tiny.yaml).",
    )
    parser.add_argument(
        "--prompt", required=True, type=str,
        help="Text prompt to seed generation.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=200,
        help="Number of new tokens to generate (default: 200).",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0).",
    )
    parser.add_argument(
        "--top-k", type=int, default=0,
        help="Top-k sampling; 0 means disabled (default: 0).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on: 'cpu', 'cuda', 'mps' (auto-detected if omitted).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # ----- Validate paths -----
    for attr, label in [
        ("checkpoint", "Checkpoint"),
        ("tokenizer_path", "Tokenizer"),
        ("config", "Config"),
    ]:
        path: Path = getattr(args, attr)
        if not path.exists():
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    # ----- Device -----
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # ----- Config & model -----
    # Import here to keep the module importable without torch installed at top level
    from src.config import ModelConfig
    from src.model import GPT

    cfg = ModelConfig.from_yaml(args.config)
    model = GPT.from_config(cfg)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Model parameters: {model.num_parameters():,}")

    # ----- Tokenizer -----
    from tokenizers import Tokenizer  # type: ignore

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))

    # ----- Encode prompt -----
    encoded = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    # ----- Generate -----
    print(f"\nPrompt: {args.prompt!r}")
    print("-" * 60)

    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    generated_ids = output_ids[0, len(encoded.ids):].tolist()
    generated_text = tokenizer.decode(generated_ids)
    print(args.prompt + generated_text)


if __name__ == "__main__":
    main()
