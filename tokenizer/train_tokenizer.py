"""Train a BPE tokenizer on daily_dialog and write the token binaries.

Run once:  python tokenizer/train_tokenizer.py
Produces:  tokenizer/vocab.json          (the tokenizer)
           data/processed/train.bin      (uint16 token ids)
           data/processed/val.bin
"""
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = ROOT / "tokenizer" / "vocab.json"
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

VOCAB_SIZE = 8192
# Chat turns are wrapped in these so the model learns who is speaking.
SPECIALS = ["<|pad|>", "<|endoftext|>", "<|user|>", "<|bot|>"]


def conversations(split):
    """Yield each dialog as one '<|user|> .. <|bot|> ..' string."""
    # OpenRL mirror: same data, Parquet instead of a loading script (datasets>=4 dropped scripts).
    ds = load_dataset("OpenRL/daily_dialog", split=split, cache_dir=str(RAW))
    for row in ds:
        turns = []
        for i, utterance in enumerate(row["dialog"]):
            speaker = "<|user|>" if i % 2 == 0 else "<|bot|>"
            turns.append(f"{speaker} {utterance.strip()}")
        yield " ".join(turns) + " <|endoftext|>"


def train_tokenizer(texts):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def encode_split(tok, split, out_path):
    ids = []
    for text in conversations(split):
        ids.extend(tok.encode(text).ids)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(out_path)
    print(f"{out_path.name}: {len(arr):,} tokens")


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print("training tokenizer...")
    train_texts = list(conversations("train"))
    tok = train_tokenizer(train_texts)
    tok.save(str(TOKENIZER_PATH))
    assert tok.get_vocab_size() <= 65535, "vocab must fit in uint16"

    # Round-trip check: the one thing that must not silently break.
    sample = "<|user|> how are you ? <|bot|> i am fine , thanks ."
    assert tok.decode(tok.encode(sample).ids).strip() == sample, "tokenizer round-trip failed"
    print(f"vocab size: {tok.get_vocab_size()}  (round-trip ok)")

    print("encoding splits...")
    encode_split(tok, "train", PROCESSED / "train.bin")
    encode_split(tok, "validation", PROCESSED / "val.bin")

    meta = {"vocab_size": tok.get_vocab_size(), "specials": SPECIALS}
    (PROCESSED / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done.")


if __name__ == "__main__":
    main()
