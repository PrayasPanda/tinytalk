# tinytalk — a conversational LLM trained from scratch

A 14M parameter decoder-only transformer, written from first principles in PyTorch
and trained on the DailyDialog corpus. No pretrained weights, no `transformers`
library — the attention, the block, the training loop and the sampler are all here.

**[💬 Try it live](https://huggingface.co/spaces/YOUR_USERNAME/tinytalk)**

## What's implemented by hand

| Piece | File |
|---|---|
| Byte-level BPE tokenizer (8k vocab) | [tokenizer/train_tokenizer.py](tokenizer/train_tokenizer.py) |
| Causal multi-head self-attention | [model/attention.py](model/attention.py) |
| Pre-norm transformer block, GELU FFN | [model/layers.py](model/layers.py) |
| Full model, weight tying, top-k sampling | [model/transformer.py](model/transformer.py) |
| Training loop, cosine LR, AMP, resume | [train/train.py](train/train.py) |
| Perplexity evaluation | [eval/metrics.py](eval/metrics.py) |

## Architecture

```
6 layers · 6 heads · 384 embedding dim · 256 context · 8192 vocab
13.9M parameters · weight-tied embeddings · learned positional embeddings
```

## Setup

```bash
pip install -r requirements.txt
```

On an RTX 50-series GPU install PyTorch with CUDA 12.8 first, or it falls back to CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Usage

```bash
# 1. Train the tokenizer and build data/processed/{train,val}.bin
python tokenizer/train_tokenizer.py

# 2. Smoke test the loop (50 steps, any machine)
python train/train.py --max-iters 50

# 3. Full training run
python train/train.py

# 4. Chat with it
python eval/generate.py

# 5. Measure it
python eval/metrics.py

# 6. Web demo
python app/app.py
```

### Training on Colab

Checkpoints are written every `eval_interval` steps and training resumes from
`latest.pt` automatically, so a disconnect costs at most one interval. Point the
checkpoint directory at Drive so they survive:

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone <this-repo> && cd llm-chatbot && pip install -r requirements.txt
!python tokenizer/train_tokenizer.py
!python train/train.py --checkpoint-dir /content/drive/MyDrive/llm-chatbot/checkpoints
```

## Verifying the components

Each core module self-checks when run directly:

```bash
python -m model.attention     # asserts the causal mask blocks future tokens
python -m model.transformer   # asserts shapes and a sane initial loss
```

## Results

Trained for 6000 iterations on a single Colab T4.

| Metric | Value |
|---|---|
| Validation loss | 2.85 (best, iter 5250) |
| Perplexity | 17.75 |
| Training time | 41 minutes |
| Tokens seen | 1.48M train / 137k val |

```
iter    0 | train 9.015 | val 9.014
iter 1000 | train 3.417 | val 3.510
iter 3000 | train 2.491 | val 2.979
iter 5250 | train 1.945 | val 2.849   <- best
iter 5999 | train 1.879 | val 2.877
```

Sample exchange:

```
you: hello , how are you ?
bot: I'm fine , thanks . We're going to the beach and I'm going to get a little late .
```

## Notes and limitations

- Trained on ~1.5M tokens of everyday dialogue, so it holds short casual
  conversations and nothing more. It is a demonstration of the architecture and
  training pipeline, not a useful assistant.
- No instruction tuning or RLHF — this is pretraining only.
- Context is 256 tokens; the chat interface keeps the last 4 turns.
- The train/val gap widens steadily after ~iter 2000 (0.05 → 0.93 by the end),
  so this run is data-bound rather than capacity-bound. More dialogue data would
  help more than more layers; `best.pt` captures the val minimum at iter 5250.
