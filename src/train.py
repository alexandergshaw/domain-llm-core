"""train.py – Training loop for domain-llm-core.

Supports:
  - CPU and CUDA (and Apple MPS) training
  - AdamW optimiser with gradient clipping
  - Configurable train/validation split
  - Best-checkpoint saving based on validation loss
  - Periodic checkpoint saves every epoch
  - Checkpoint resume (--resume)
  - Periodic sample generation during training
  - CLI arguments for all key paths and hyper-parameters

Quick start
-----------
    python src/train.py \\
        --config configs/tiny.yaml \\
        --tokenizer-path tokenizer.json \\
        --train-file data/train.txt \\
        --checkpoint-dir checkpoints/
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a decoder-only GPT model from scratch."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tiny.yaml"),
        help="Path to YAML config file (default: configs/tiny.yaml).",
    )
    parser.add_argument(
        "--tokenizer-path", type=Path, default=Path("tokenizer.json"),
        help="Path to a HuggingFace tokenizers JSON file.",
    )
    parser.add_argument(
        "--train-file", type=Path, default=Path("data/train.txt"),
        help="Path to the raw training text file.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints"),
        help="Directory where checkpoints are saved (default: checkpoints/).",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.05,
        help="Fraction of data used for validation (default: 0.05).",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="Gradient clipping max norm (default: 1.0).",
    )
    parser.add_argument(
        "--sample-every", type=int, default=0,
        help="Generate a sample every N steps during training. 0 = disabled.",
    )
    parser.add_argument(
        "--sample-prompt", type=str, default="The",
        help="Prompt for periodic sample generation.",
    )
    parser.add_argument(
        "--sample-tokens", type=int, default=80,
        help="Number of tokens for periodic samples (default: 80).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override: 'cpu', 'cuda', 'mps'. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="DataLoader num_workers (default: 0).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(path: Path):
    """Load a HuggingFace tokenizers JSON file.  Exits with a clear message if missing."""
    if not path.exists():
        print(
            f"ERROR: Tokenizer not found at '{path}'.\n"
            f"  Train a BPE tokenizer first and save it to '{path}', or\n"
            f"  pass --tokenizer-path to point at an existing tokenizer.json.",
            file=sys.stderr,
        )
        sys.exit(1)
    from tokenizers import Tokenizer  # type: ignore
    return Tokenizer.from_file(str(path))


def load_text(path: Path) -> str:
    """Read training text.  Exits with a clear message if the file is missing."""
    if not path.exists():
        print(
            f"ERROR: Training file not found at '{path}'.\n"
            f"  Place your training corpus at '{path}', or\n"
            f"  pass --train-file to specify a different location.",
            file=sys.stderr,
        )
        sys.exit(1)
    return path.read_text(encoding="utf-8")


@torch.inference_mode()
def sample_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Generate a short text sample for progress monitoring."""
    from src.generate import generate  # local import to avoid circular deps

    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    output_ids = generate(model, input_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    generated = output_ids[0, len(encoded.ids):].tolist()
    model.train()
    return prompt + tokenizer.decode(generated)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.CrossEntropyLoss, device: torch.device) -> float:
    """Compute average cross-entropy loss over a DataLoader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), target_ids.view(B * T))
            total_loss += loss.item() * B * T
            total_tokens += B * T
    model.train()
    return total_loss / total_tokens if total_tokens else float("inf")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_args(argv)

    # ---- Imports (inside main so the module stays importable for tests) ----
    from src.config import ModelConfig
    from src.model import GPT
    from src.dataset import TokenDataset

    # ---- Validate config file ----
    if not args.config.exists():
        print(
            f"ERROR: Config file not found at '{args.config}'.\n"
            f"  Pass --config to specify the YAML config (e.g. configs/tiny.yaml).",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = ModelConfig.from_yaml(args.config)
    print(cfg)

    # ---- Device ----
    device = select_device(args.device)
    print(f"Using device: {device}")

    # ---- Checkpoint directory ----
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = load_tokenizer(args.tokenizer_path)

    # ---- Training text ----
    text = load_text(args.train_file)
    print(f"Loaded {len(text):,} characters from '{args.train_file}'.")

    encoded = tokenizer.encode(text)
    token_ids = encoded.ids
    print(f"Tokenized to {len(token_ids):,} tokens.")

    # ---- Dataset & split ----
    full_dataset = TokenDataset(token_ids, context_length=cfg.context_length)

    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    if train_size <= 0:
        print(
            f"ERROR: Dataset too small ({len(full_dataset)} samples) for a "
            f"{args.val_split:.0%} validation split.",
            file=sys.stderr,
        )
        sys.exit(1)

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train samples: {train_size:,}  |  Val samples: {val_size:,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    model = GPT.from_config(cfg).to(device)
    print(f"Model parameters: {model.num_parameters():,}")

    # ---- Optimiser ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.1,
    )

    criterion = nn.CrossEntropyLoss()

    # ---- Resume from checkpoint ----
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        if not args.resume.exists():
            print(f"ERROR: Resume checkpoint not found: '{args.resume}'", file=sys.stderr)
            sys.exit(1)
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from '{args.resume}' at epoch {start_epoch}.")

    # ---- Training ----
    model.train()
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, cfg.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for step, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)               # (B, T, V)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), target_ids.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item() * B * T
            epoch_tokens += B * T
            global_step += 1

            # ---- Periodic sample generation ----
            if args.sample_every > 0 and global_step % args.sample_every == 0:
                sample = sample_text(
                    model, tokenizer, args.sample_prompt,
                    args.sample_tokens, device,
                )
                print(f"\n[step {global_step}] Sample:\n{sample}\n")

            # ---- Progress printing (every 10 steps or last step) ----
            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                avg_loss = epoch_loss / epoch_tokens
                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch + 1}/{cfg.epochs}  "
                    f"Step {step + 1}/{len(train_loader)}  "
                    f"Loss: {avg_loss:.4f}  "
                    f"Perplexity: {math.exp(avg_loss):.2f}  "
                    f"({elapsed:.1f}s)"
                )

        # ---- Epoch summary ----
        train_loss = epoch_loss / epoch_tokens
        val_loss = evaluate(model, val_loader, criterion, device)
        print(
            f"\n{'=' * 60}\n"
            f"Epoch {epoch + 1} complete  "
            f"Train loss: {train_loss:.4f}  "
            f"Val loss: {val_loss:.4f}  "
            f"Val ppl: {math.exp(val_loss):.2f}\n"
            f"{'=' * 60}\n"
        )

        # ---- Save epoch checkpoint ----
        ckpt_path = args.checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "config": cfg.__dict__,
            },
            ckpt_path,
        )
        print(f"Checkpoint saved: {ckpt_path}")

        # ---- Save best checkpoint ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = args.checkpoint_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "config": cfg.__dict__,
                },
                best_path,
            )
            print(f"New best checkpoint saved: {best_path}  (val_loss={val_loss:.4f})")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
