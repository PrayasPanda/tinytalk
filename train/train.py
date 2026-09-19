"""Training loop. Resumes automatically — Colab will disconnect on you.

Local smoke test:  python train/train.py --max-iters 50
Colab:             python train/train.py
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import GPT  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def load_config(overrides):
    cfg = yaml.safe_load((ROOT / "train" / "config.yaml").read_text())
    if overrides.max_iters:
        cfg["train"]["max_iters"] = overrides.max_iters
    if overrides.checkpoint_dir:
        cfg["checkpoint_dir"] = overrides.checkpoint_dir
    return cfg


def load_split(name):
    path = PROCESSED / f"{name}.bin"
    if not path.exists():
        sys.exit(f"missing {path} — run: python tokenizer/train_tokenizer.py")
    return np.fromfile(path, dtype=np.uint16)


def get_batch(data, batch_size, block_size, device):
    """Random windows of the token stream; targets are inputs shifted by one."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, splits, cfg, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg["train"]["eval_iters"])
        for i in range(cfg["train"]["eval_iters"]):
            x, y = get_batch(data, cfg["train"]["batch_size"], cfg["model"]["block_size"], device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(step, tcfg):
    """Linear warmup, then cosine decay to min_lr."""
    if step < tcfg["warmup_iters"]:
        return tcfg["learning_rate"] * (step + 1) / tcfg["warmup_iters"]
    progress = (step - tcfg["warmup_iters"]) / max(1, tcfg["max_iters"] - tcfg["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return tcfg["min_lr"] + coeff * (tcfg["learning_rate"] - tcfg["min_lr"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, help="override max_iters (use 50 for a smoke test)")
    ap.add_argument("--checkpoint-dir", help="override checkpoint dir (point at Drive on Colab)")
    args = ap.parse_args()

    cfg = load_config(args)
    tcfg = cfg["train"]
    torch.manual_seed(tcfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    splits = {"train": load_split("train"), "val": load_split("val")}
    vocab_size = int(max(splits["train"].max(), splits["val"].max())) + 1

    model = GPT(vocab_size=vocab_size, **cfg["model"]).to(device)
    print(f"params: {model.num_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"]
    )
    # bf16 on modern GPUs, fp16 elsewhere; CPU stays fp32.
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device, enabled=use_amp and amp_dtype is torch.float16)

    ckpt_dir = ROOT / cfg["checkpoint_dir"] if not Path(cfg["checkpoint_dir"]).is_absolute() else Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"

    start_iter, best_val = 0, float("inf")
    if latest.exists():
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_iter, best_val = ck["iter"] + 1, ck["best_val"]
        print(f"resumed from iter {start_iter}")

    t0 = time.time()
    for step in range(start_iter, tcfg["max_iters"]):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, tcfg)

        x, y = get_batch(splits["train"], tcfg["batch_size"], cfg["model"]["block_size"], device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if step % tcfg["eval_interval"] == 0 or step == tcfg["max_iters"] - 1:
            losses = estimate_loss(model, splits, cfg, device)
            elapsed = time.time() - t0
            print(
                f"iter {step:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
                f"| lr {lr_at(step, tcfg):.2e} | {elapsed/60:.1f}m"
            )

            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": step,
                "best_val": min(best_val, losses["val"]),
                "config": cfg,
                "vocab_size": vocab_size,
            }
            torch.save(payload, latest)
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(payload, ckpt_dir / "best.pt")

    print(f"done. best val loss {best_val:.3f}, checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
