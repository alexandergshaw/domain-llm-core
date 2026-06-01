# domain-llm-core

A minimal, from-scratch decoder-only transformer (GPT-style) trained with PyTorch.
The goal is a clean, well-commented reference implementation of a small language model
that can be trained on any plain-text corpus on a single GPU or CPU.

---

## Project goals

- Implement every component of a GPT-style model from first principles
- Keep the codebase small enough to read in an afternoon
- Support practical local training: checkpoint resume, validation loss, best-model saving
- Produce readable generated text on small corpora within a few hours of GPU time

---

## Repository structure

```
domain-llm-core/
├── src/
│   ├── config.py      – YAML → dataclass configuration loader
│   ├── model.py       – Decoder-only transformer (attention, FFN, embeddings)
│   ├── dataset.py     – Sliding-window next-token prediction dataset
│   ├── train.py       – Full training loop with validation + checkpointing
│   └── generate.py    – Autoregressive text generation from a checkpoint
├── configs/
│   └── tiny.yaml      – Tiny model config (~15 M params)
├── data/              – Place train.txt here (gitignored, tracked via .gitkeep)
├── checkpoints/       – Saved checkpoints (gitignored, tracked via .gitkeep)
├── tests/             – Pytest unit tests
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/alexandergshaw/domain-llm-core.git
cd domain-llm-core

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Prepare your data & tokenizer

You need two files before training:

| File | Description |
|---|---|
| `data/train.txt` | Raw training text (UTF-8) |
| `tokenizer.json` | A HuggingFace *tokenizers* BPE tokenizer |

### Train a BPE tokenizer (example)

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=8000, special_tokens=["[UNK]"])
tokenizer.train(files=["data/train.txt"], trainer=trainer)
tokenizer.save("tokenizer.json")
```

---

## Training

```bash
python src/train.py \
    --config configs/tiny.yaml \
    --tokenizer-path tokenizer.json \
    --train-file data/train.txt \
    --checkpoint-dir checkpoints/
```

### Key CLI options

| Flag | Default | Description |
|---|---|---|
| `--config` | `configs/tiny.yaml` | Path to YAML model config |
| `--tokenizer-path` | `tokenizer.json` | HuggingFace tokenizers JSON |
| `--train-file` | `data/train.txt` | Raw training corpus |
| `--checkpoint-dir` | `checkpoints/` | Where to save checkpoints |
| `--resume` | *(none)* | Checkpoint to resume from |
| `--val-split` | `0.05` | Fraction of data held out for validation |
| `--max-grad-norm` | `1.0` | Gradient clipping norm |
| `--sample-every` | `0` | Generate a sample every N steps (0 = off) |
| `--sample-prompt` | `"The"` | Prompt used for periodic samples |
| `--device` | auto | Force `cpu`, `cuda`, or `mps` |

### Resume training from a checkpoint

```bash
python src/train.py \
    --config configs/tiny.yaml \
    --tokenizer-path tokenizer.json \
    --train-file data/train.txt \
    --checkpoint-dir checkpoints/ \
    --resume checkpoints/epoch_003.pt
```

---

## Text generation

```bash
python src/generate.py \
    --checkpoint checkpoints/best.pt \
    --tokenizer-path tokenizer.json \
    --config configs/tiny.yaml \
    --prompt "Once upon a time" \
    --max-new-tokens 200 \
    --temperature 0.8 \
    --top-k 40
```

### Key CLI options

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to model checkpoint |
| `--tokenizer-path` | *(required)* | HuggingFace tokenizers JSON |
| `--config` | *(required)* | YAML model config |
| `--prompt` | *(required)* | Seed text for generation |
| `--max-new-tokens` | `200` | Number of tokens to generate |
| `--temperature` | `1.0` | Sampling temperature |
| `--top-k` | `0` | Top-k sampling (0 = disabled) |

---

## Model configuration (`configs/tiny.yaml`)

```yaml
vocab_size: 8000
context_length: 512
layers: 6
hidden_size: 384
attention_heads: 6
dropout: 0.1
batch_size: 16
learning_rate: 0.0003
epochs: 5
```

Approximate parameter count for the tiny config: **~15 M parameters**.

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Architecture overview

```
Input token IDs
      │
      ├── TokenEmbedding  (vocab_size → hidden_size)
      └── PositionalEmbedding  (context_length → hidden_size)
                │
          Embedding Dropout
                │
    ┌──── TransformerBlock × N ────┐
    │   Pre-LayerNorm               │
    │   CausalSelfAttention         │   ← upper-triangular causal mask
    │   Residual add                │
    │   Pre-LayerNorm               │
    │   FeedForward MLP (4× hidden) │
    │   Residual add                │
    └───────────────────────────────┘
                │
          Final LayerNorm
          Linear projection → vocab_size logits
```

Weight tying: the output projection shares weights with the token embedding,
reducing parameters and improving perplexity.
