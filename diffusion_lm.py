from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- model ---
class DiffusionLM(nn.Module):
    """Small transformer trained to reverse an absorbing (mask) corruption.

    MASK occupies index vocab_size, one past the real vocabulary, so no real
    symbol is displaced and a model trained on one dataset can be inspected
    against another with the same alphabet.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 4, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = vocab_size
        self.tok = nn.Embedding(vocab_size + 1, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.time = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x, t):
        """x: [B, L] token ids with MASK where corrupted. t: [B] in (0, 1]."""
        L = x.shape[1]
        idx = torch.arange(L, device=x.device)
        h = self.tok(x) + self.pos(idx)[None]
        h = h + self.time(t[:, None, None].float())
        # pad positions carry no information and are excluded from attention
        return self.out(self.norm(self.enc(h, src_key_padding_mask=(x == 0))))


# ------------------------------------------------------------------ training ---
def corrupt(x, t, mask_id, gen=None):
    """Absorbing forward process: mask each non-pad token with probability t."""
    r = torch.rand(x.shape, device=x.device, generator=gen)
    m = (r < t[:, None]) & (x != 0)
    return torch.where(m, torch.full_like(x, mask_id), x), m


def elbo_terms(model, x, t, gen=None):
    """Per-sequence negative ELBO contribution at time t.

    For absorbing diffusion the continuous-time bound is

        -ELBO = E_t [ (1/t) * sum_{masked} -log p(x_i | x_t) ]

    so cross entropies at low t, where few tokens are masked and the task is
    easy, are up-weighted. Returned per sequence and normalised by the number
    of real tokens, so sequences of different padding are comparable.
    """
    xt, m = corrupt(x, t, model.mask_id, gen)
    logits = model(xt, t)
    ce = F.cross_entropy(logits.reshape(-1, model.vocab_size),
                         x.reshape(-1), reduction="none").reshape(x.shape)
    ce = ce * m.float()
    n_real = (x != 0).sum(1).clamp(min=1).float()
    return (ce.sum(1) / t.clamp(min=1e-3)) / n_real


def train(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    V = int(meta["cipher"]["vocab_size"])
    # The plaintext bank only. The model never sees ciphertext, which is what
    # keeps the attack unsupervised.
    data = torch.tensor(d["train_plain"], dtype=torch.long)
    print(f"[diffusion] {data.shape[0]} plaintext samples, length "
          f"{data.shape[1]}, vocab {V}, device {dev}")

    model = DiffusionLM(V, d_model=args.d_model, n_layers=args.n_layers,
                        max_len=max(256, data.shape[1])).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[diffusion] {n_par/1e6:.2f} M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)

    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, data.shape[0], (args.batch_size,))
        x = data[idx].to(dev)
        t = torch.rand(x.shape[0], device=dev).clamp(min=1e-3)
        loss = elbo_terms(model, x, t).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % args.print_every == 0 or step == 1:
            print(f"  step {step:6d}/{args.steps}  -ELBO/token {loss.item():8.4f}"
                  f"  {time.time()-t0:6.0f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state": model.state_dict(), "vocab_size": V,
                "d_model": args.d_model, "n_layers": args.n_layers,
                "max_len": max(256, data.shape[1])}, args.out)
    print(f"wrote {args.out}")


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = DiffusionLM(ck["vocab_size"], d_model=ck["d_model"],
                    n_layers=ck["n_layers"], max_len=ck["max_len"]).to(dev)
    m.load_state_dict(ck["state"]); m.eval()
    return m


# ------------------------------------------------------------------- scoring ---
@torch.no_grad()
def score(model, x, n_t: int = 8, batch: int = 256, seed: int = 0,
          dev=None) -> float:
    """Mean negative ELBO per token. Lower is more English-like.

    Common random numbers: the timesteps and the mask pattern are drawn from a
    fixed seed, so two candidate decryptions are compared under identical
    corruption. Without this the estimator noise swamps the difference between
    two similar keys, and any search using the score wanders.
    """
    dev = dev or next(model.parameters()).device
    x = torch.as_tensor(x, dtype=torch.long)
    gen = torch.Generator(device=dev).manual_seed(seed)
    ts = torch.linspace(0.05, 0.95, n_t, device=dev)
    total, n = 0.0, 0
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch].to(dev)
        for t in ts:
            tt = t.expand(xb.shape[0])
            g = torch.Generator(device=dev).manual_seed(seed + int(t * 1e6))
            total += elbo_terms(model, xb, tt, g).sum().item()
            n += xb.shape[0]
    return total / max(n, 1)


def bigram_score(x, plain_bank, V, alpha=0.5) -> float:
    """The classical attack's scoring function, for comparison."""
    c = np.zeros((V, V))
    a, b = plain_bank[:, :-1].ravel(), plain_bank[:, 1:].ravel()
    keep = (a != 0) & (b != 0)
    np.add.at(c, (a[keep], b[keep]), 1.0)
    c += alpha
    L = np.log(c / c.sum(1, keepdims=True))
    a, b = x[:, :-1].ravel(), x[:, 1:].ravel()
    keep = (a != 0) & (b != 0)
    return float(L[a[keep], b[keep]].mean())


# ------------------------------------------------------- the go/no-go test ---
def discriminate(args):
    """Does the diffusion model separate good decryptions from bad ones better
    than the bigram model does?

    Candidates are generated by degrading the true plaintext through a known
    number of key errors, which gives a ground-truth ordering to check the two
    scores against. If diffusion does not rank more faithfully than bigrams,
    nothing built on top of it will help, and the rest of the plan should be
    abandoned rather than debugged.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    V = int(meta["cipher"]["vocab_size"])
    plain = d["eval_plain"]
    bank = d["train_plain"]
    model = load_model(args.model, dev)
    rng = np.random.default_rng(0)

    sub = plain[:args.n_eval]
    rows = []
    for n_err in args.errors:
        # a key with exactly n_err symbols permuted away from the truth
        key = np.arange(V)
        if n_err >= 2:
            pick = rng.permutation(np.arange(1, V))[:n_err]
            key[pick] = key[np.roll(pick, 1)]
        cand = key[sub]
        cand[sub == 0] = 0
        acc = float((cand == sub).mean())
        rows.append({
            "errors": n_err, "accuracy": acc,
            "diffusion": score(model, cand, n_t=args.n_t, seed=0, dev=dev),
            "bigram": bigram_score(cand, bank, V),
        })
        print(f"  {n_err:3d} key errors  acc {acc:.4f}  "
              f"diffusion {rows[-1]['diffusion']:9.4f}  "
              f"bigram {rows[-1]['bigram']:8.4f}")

    accs = np.array([r["accuracy"] for r in rows])
    dif = -np.array([r["diffusion"] for r in rows])   # higher = better
    big = np.array([r["bigram"] for r in rows])

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    def dyn_range(s):
        """Score gap between perfect and worst, in units of its own spread."""
        return float((s.max() - s.min()) / (np.std(s) + 1e-9))

    print("\n" + "=" * 64)
    print(f"rank correlation with true accuracy")
    print(f"  diffusion  {spearman(accs, dif):+.4f}")
    print(f"  bigram     {spearman(accs, big):+.4f}")
    print(f"separation between best and worst candidate")
    print(f"  diffusion  {dyn_range(dif):.3f}")
    print(f"  bigram     {dyn_range(big):.3f}")
    print("=" * 64)
    print("The diffusion model is worth building on only if its rank")
    print("correlation is at least as high as the bigram model's. If it is")
    print("lower, stop here: the search cannot be better than its score.")

    if args.out:
        json.dump({"rows": rows,
                   "spearman_diffusion": spearman(accs, dif),
                   "spearman_bigram": spearman(accs, big)},
                  open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--npz_path", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch_size", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--d_model", type=int, default=256)
    t.add_argument("--n_layers", type=int, default=4)
    t.add_argument("--print_every", type=int, default=500)
    t.set_defaults(func=train)

    s = sub.add_parser("discriminate")
    s.add_argument("--npz_path", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--n_eval", type=int, default=512)
    s.add_argument("--n_t", type=int, default=8)
    s.add_argument("--errors", type=int, nargs="+",
                   default=[0, 2, 4, 6, 10, 14, 20, 26])
    s.add_argument("--out", default=None)
    s.set_defaults(func=discriminate)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
