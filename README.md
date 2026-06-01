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
│   ├── config.py                    – YAML → dataclass configuration loader
│   ├── model.py                     – Decoder-only transformer (attention, FFN, embeddings)
│   ├── dataset.py                   – Sliding-window next-token prediction dataset
│   ├── train.py                     – Full training loop with validation + checkpointing
│   ├── generate.py                  – Autoregressive text generation from a checkpoint
│   └── prepare_instruction_data.py  – Convert JSONL instruction examples → train.txt
├── configs/
│   └── tiny.yaml      – Tiny model config (~15 M params)
├── examples/
│   └── instruction_examples.jsonl   – Sample instruction-tuning examples
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

## Instruction-tuning workflow

You can fine-tune (or train from scratch) on instruction-following data using
`src/prepare_instruction_data.py`.

### JSONL format

Each line of the examples file must be a JSON object with these fields:

| Field | Required non-empty | Description |
|---|---|---|
| `instruction` | ✓ | The task description |
| `input` | ✗ (may be empty) | Optional additional context |
| `output` | ✓ | The expected model response |
| `source_file` | ✓ | Provenance / origin filename |
| `category` | ✓ | Task category (used for summary counts) |

Example record:

```json
{"instruction": "Summarize this.", "input": "Some text here.", "output": "A concise summary.", "source_file": "my_data.txt", "category": "summarization"}
```

Records with missing or wrong-typed required fields are **skipped with a warning**.
An empty `input` field is **allowed** — many tasks need no additional context.

### Formatted output

Each valid record is written to the output file in this format:

```
<bos>
### Instruction:
<instruction text>

### Input:
<input text>

### Response:
<output text>
<eos>
```

Records are separated by a blank line.

### Prepare training data

```bash
# Use the bundled sample examples
python src/prepare_instruction_data.py \
    --examples-file examples/instruction_examples.jsonl \
    --output-file data/train.txt

# Or point at your own JSONL file
python src/prepare_instruction_data.py \
    --examples-file /path/to/my_examples.jsonl \
    --output-file data/train.txt
```

The script prints a per-category summary:

```
Processed 10 rows from 'examples/instruction_examples.jsonl'
  Valid   : 10
  Skipped :  0

Examples by category:
  classification    1
  code_generation   2
  question_answering 3
  reasoning         1
  summarization     1
  text_editing      1
  translation       1

Output written to 'data/train.txt' (10 examples).
```

### Train a tokenizer, then train the model

```bash
# 1. Prepare the text
python src/prepare_instruction_data.py \
    --examples-file examples/instruction_examples.jsonl \
    --output-file data/train.txt

# 2. Train a BPE tokenizer on the prepared text
python - <<'EOF'
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tok = Tokenizer(BPE(unk_token="[UNK]"))
tok.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=8000, special_tokens=["[UNK]", "<bos>", "<eos>"])
tok.train(files=["data/train.txt"], trainer=trainer)
tok.save("tokenizer.json")
EOF

# 3. Train the model
python src/train.py \
    --config configs/tiny.yaml \
    --tokenizer-path tokenizer.json \
    --train-file data/train.txt \
    --checkpoint-dir checkpoints/

# 4. Generate a response
python src/generate.py \
    --checkpoint checkpoints/best.pt \
    --tokenizer-path tokenizer.json \
    --config configs/tiny.yaml \
    --prompt $'<bos>\n### Instruction:\nExplain what dropout is.\n\n### Input:\n\n### Response:\n' \
    --max-new-tokens 150 \
    --temperature 0.7 \
    --top-k 40
```

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
