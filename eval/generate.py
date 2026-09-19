"""Chat with a trained checkpoint from the terminal.

    python eval/generate.py                      # interactive
    python eval/generate.py --prompt "hello"     # one shot
"""
import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import GPT  # noqa: E402


def load(checkpoint, device):
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.exists():
        sys.exit(f"no checkpoint at {ckpt_path} — train first, or pass --checkpoint")

    ck = torch.load(ckpt_path, map_location=device)
    model = GPT(vocab_size=ck["vocab_size"], **ck["config"]["model"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    tok = Tokenizer.from_file(str(ROOT / "tokenizer" / "vocab.json"))
    return model, tok


def reply(model, tok, history, device, max_new_tokens=80, temperature=0.8, top_k=40):
    """history: list of (user, bot) pairs; the last bot may be None."""
    parts = []
    for user, bot in history:
        parts.append(f"<|user|> {user}")
        if bot is not None:
            parts.append(f"<|bot|> {bot}")
    parts.append("<|bot|>")
    prompt = " ".join(parts)

    ids = tok.encode(prompt).ids
    # Leave room for what we are about to generate.
    ids = ids[-(model.block_size - max_new_tokens):]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    stop = tok.token_to_id("<|user|>")
    out = model.generate(idx, max_new_tokens, temperature, top_k, stop_token=stop)
    text = tok.decode(out[0, len(ids):].tolist())
    # Cut at the next speaker tag if the model rambles into one.
    for tag in ("<|user|>", "<|bot|>", "<|endoftext|>"):
        text = text.split(tag)[0]
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="train/checkpoints/best.pt")
    ap.add_argument("--prompt", help="single prompt instead of interactive mode")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load(args.checkpoint, device)

    if args.prompt:
        print(reply(model, tok, [(args.prompt, None)], device, temperature=args.temperature, top_k=args.top_k))
        return

    print("chat (ctrl-c to quit)")
    history = []
    while True:
        try:
            user = input("you: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not user:
            continue
        answer = reply(model, tok, history + [(user, None)], device,
                       temperature=args.temperature, top_k=args.top_k)
        print(f"bot: {answer}")
        history.append((user, answer))
        history = history[-4:]  # keep the context window from overflowing


if __name__ == "__main__":
    main()
