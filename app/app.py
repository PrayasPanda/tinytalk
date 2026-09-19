"""Gradio chat demo. Runs locally and on a Hugging Face Space unchanged.

    python app/app.py

On a Space, commit checkpoint to train/checkpoints/best.pt (git-lfs) and
tokenizer/vocab.json alongside this file.
"""
import sys
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.generate import load, reply  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, TOKENIZER = load("train/checkpoints/best.pt", DEVICE)


def chat(message, history, temperature, top_k):
    # Gradio gives history as [{"role": ..., "content": ...}]; fold into pairs.
    pairs = []
    for turn in history:
        if turn["role"] == "user":
            pairs.append((turn["content"], None))
        elif pairs:
            pairs[-1] = (pairs[-1][0], turn["content"])
    pairs.append((message, None))

    return reply(MODEL, TOKENIZER, pairs[-4:], DEVICE,
                 temperature=temperature, top_k=int(top_k))


demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="Small GPT — trained from scratch",
    description=(
        f"A {MODEL.num_params()/1e6:.0f}M parameter transformer written and trained from "
        "scratch on the DailyDialog corpus. No pretrained weights."
    ),
    additional_inputs=[
        gr.Slider(0.1, 1.5, value=0.8, label="temperature"),
        gr.Slider(1, 100, value=40, step=1, label="top-k"),
    ],
    examples=["hello , how are you ?", "what do you do on weekends ?", "i am going to the movies"],
)

if __name__ == "__main__":
    demo.launch()
