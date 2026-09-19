"""The full GPT: embeddings -> N blocks -> logits, plus sampling."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Block


class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: input and output embeddings share one matrix.
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence of {T} exceeds block_size {self.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, stop_token=None):
        """Sample tokens one at a time, feeding each back in."""
        self.eval()
        for _ in range(max_new_tokens):
            # Never feed more than block_size of context.
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k)[0][:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            if stop_token is not None and (next_id == stop_token).all():
                break
        return idx


def _self_check():
    torch.manual_seed(0)
    model = GPT(vocab_size=100, n_layer=2, n_head=2, n_embd=32, block_size=16)
    idx = torch.randint(0, 100, (2, 16))

    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 16, 100), logits.shape
    # Untrained loss should sit near ln(vocab_size) = 4.6.
    assert 3.0 < loss.item() < 6.5, f"suspicious initial loss: {loss.item()}"

    # Generation must respect block_size even when asked to exceed it.
    out = model.generate(idx[:, :4], max_new_tokens=20)
    assert out.shape == (2, 24), out.shape
    print(f"model ok: loss={loss.item():.2f}, params={model.num_params():,}")


if __name__ == "__main__":
    _self_check()
