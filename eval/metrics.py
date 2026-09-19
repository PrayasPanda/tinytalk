"""Validation loss and perplexity for a checkpoint.

    python eval/metrics.py
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import GPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="train/checkpoints/best.pt")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--batches", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = ROOT / args.checkpoint
    if not ckpt_path.exists():
        sys.exit(f"no checkpoint at {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device)

    model = GPT(vocab_size=ck["vocab_size"], **ck["config"]["model"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    data = np.fromfile(ROOT / "data" / "processed" / f"{args.split}.bin", dtype=np.uint16)
    block_size = ck["config"]["model"]["block_size"]
    batch_size = ck["config"]["train"]["batch_size"]

    torch.manual_seed(0)  # same batches every run, so numbers are comparable
    losses = []
    with torch.no_grad():
        for _ in range(args.batches):
            ix = torch.randint(len(data) - block_size - 1, (batch_size,))
            x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]).to(device)
            y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]).to(device)
            _, loss = model(x, y)
            losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    print(f"split:      {args.split}")
    print(f"loss:       {mean_loss:.4f}")
    print(f"perplexity: {math.exp(mean_loss):.2f}")
    print(f"trained to: iter {ck['iter']}")


if __name__ == "__main__":
    main()
