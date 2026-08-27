"""
select_checkpoint.py
====================

Choose which training checkpoint to report, without using the evaluation
labels.

    python select_checkpoint.py --name subst_s0 \\
        --npz_path data/substitution/dataset.npz

WHY THIS IS NEEDED
------------------
Both identity arms peaked around epoch 20 and then degraded: arm_nogp from
1.0000 to 0.7514, arm_wgangp from 1.0000 to 0.9977. So the final checkpoint is
not the one to report. But picking the best epoch by looking at character
accuracy uses the aligned eval split, and an attacker holding only ciphertext
does not have that. Reporting a peak chosen that way is selection on the test
set, and it inflates every number in the table.

THE CRITERION
-------------
Score each checkpoint's decryption by its log-likelihood under a bigram model
of English fitted on the unpaired plaintext bank, then select the checkpoint
whose score is CLOSEST TO the likelihood real plaintext has under that model.

Not the highest. An earlier version of this script maximised the likelihood and
chose the wrong checkpoint on both identity arms: it took epoch 25 at 0.9767
over epoch 20 at 1.0000, and epoch 40 at 0.9612 over epoch 15 at 1.0000. The
reason is that a correct decryption reproduces English exactly, and English has
a specific bigram likelihood - about -2.343 on this corpus, which is what every
perfect checkpoint scores. A model can beat that by over-producing frequent
bigrams, so maximising the score rewards smoothing towards common patterns
rather than accuracy. The reference value is the target, not the ceiling.

The reference is measured, not assumed: it is the likelihood the unpaired
plaintext bank itself achieves under the model. No evaluation data is involved.

Residual error in the reference is about 0.0007 nats, which on these runs costs
at most 0.0008 character accuracy - roughly eight characters in ten thousand.
Below that the criterion cannot distinguish checkpoints, and the write-up
should say so rather than presenting near-ties as discriminations.

This is exactly the self-verification that makes the classical attack
legitimate. A cryptanalyst runs several chains and keeps the one whose output
reads like English; they do not need the answer key to know which chain won.
Applying the same rule to checkpoints puts the two methods on equal footing.

WHAT IT NEVER TOUCHES
---------------------
eval_plain. The dumped pred_<epoch>.npz files contain only the model's output
and the ciphertext it was given. The language model is fitted on train_plain,
which under --split_mode disjoint shares no samples with the eval split;
mcmc_baseline.py --audit verifies this.

REPORTING HONESTLY
------------------
With --show_accuracy the table also prints the true character accuracy of each
checkpoint, read from eval_metrics_<epoch>.json. That column plays no part in
the choice. It is there so the write-up can state how often unsupervised
selection landed on the accuracy-optimal checkpoint, and what it cost when it
did not. Quote the accuracy of the SELECTED row as the result; quote the gap to
the best row as a limitation.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np


def bigram_logprobs(plain: np.ndarray, V: int, alpha: float = 0.5) -> np.ndarray:
    """Add-alpha bigram log-probabilities, fitted on the plaintext bank."""
    counts = np.zeros((V, V), dtype=np.float64)
    a, b = plain[:, :-1].ravel(), plain[:, 1:].ravel()
    keep = (a != 0) & (b != 0)
    np.add.at(counts, (a[keep], b[keep]), 1.0)
    counts += alpha
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def score(pred: np.ndarray, L: np.ndarray) -> float:
    """Mean log-probability per bigram of a decryption. Higher is better."""
    a, b = pred[:, :-1].ravel(), pred[:, 1:].ravel()
    keep = (a != 0) & (b != 0)
    if not keep.any():
        return float("-inf")
    return float(L[a[keep], b[keep]].mean())


def entropy_of(pred: np.ndarray, V: int) -> float:
    """Unigram entropy in bits. A collapsed generator scores low here.

    Diagnostic only, never part of the selection: a model that maps everything
    to one symbol can still be caught by the bigram score, but the entropy
    makes the failure mode obvious at a glance.
    """
    c = np.bincount(pred[pred != 0].ravel(), minlength=V).astype(np.float64)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def epoch_key(e: str):
    """Sort numeric epochs numerically, and keep 'latest' at the end."""
    return (1, 0) if not e.isdigit() else (0, int(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name under checkpoints/")
    ap.add_argument("--npz_path", required=True, help="the dataset it trained on")
    ap.add_argument("--checkpoints_dir", default="./checkpoints")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--show_accuracy", action="store_true",
                    help="print true accuracy alongside; NOT used to select")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    V = int(meta["cipher"]["vocab_size"])

    # Fit on the whole plaintext bank and read the reference off the same
    # data. Measured against a known-perfect decryption, that lands within
    # 0.0007 of the target, where a held-out half lands 0.0014 away and 5-fold
    # CV 0.0010 away with fold sd of 0.0014. With ~2.5M bigrams over 784
    # parameters the in-sample bias is far smaller than the variance cost of
    # discarding half the corpus.
    tp = d["train_plain"]
    L = bigram_logprobs(tp, V, alpha=args.alpha)
    reference = score(tp, L)

    run_dir = os.path.join(args.checkpoints_dir, args.name)
    files = glob.glob(os.path.join(run_dir, "pred_*.npz"))
    if not files:
        raise SystemExit(
            f"no pred_*.npz in {run_dir}\n"
            "Apply patch_dump_pred.py, then re-run eval_cipher.py for each "
            "epoch you want considered.")

    rows = []
    for f in files:
        e = re.match(r"pred_(.+)\.npz$", os.path.basename(f)).group(1)
        p = np.load(f)["pred"].astype(np.int64)
        row = {"epoch": e, "logp": score(p, L), "entropy": entropy_of(p, V)}
        if args.show_accuracy:
            mf = os.path.join(run_dir, f"eval_metrics_{e}.json")
            if os.path.exists(mf):
                m = json.load(open(mf))
                row["accuracy"] = m.get("character_accuracy")
                row["key"] = m.get("key_recovery_accuracy")
        rows.append(row)

    for r in rows:
        r["gap"] = abs(r["logp"] - reference)
    rows.sort(key=lambda r: epoch_key(r["epoch"]))
    # ties break to the earliest epoch, deterministically
    best = min(rows, key=lambda r: (r["gap"], epoch_key(r["epoch"])))

    hdr = f"{'epoch':>8s} {'logP/bigram':>12s} {'|gap|':>8s} {'entropy':>8s}"
    if args.show_accuracy:
        hdr += f" {'char acc':>9s} {'key':>7s}"
    print(f"\nrun {args.name}   LM fitted on {len(tp)} plaintext samples; "
          f"reference logP/bigram {reference:.4f}")
    print(hdr)
    for r in rows:
        line = (f"{r['epoch']:>8s} {r['logp']:12.4f} {r['gap']:8.4f} "
                f"{r['entropy']:8.3f}")
        if args.show_accuracy:
            a, k = r.get("accuracy"), r.get("key")
            line += (f" {a:9.4f} {k:7.4f}" if a is not None
                     else f" {'-':>9s} {'-':>7s}")
        if r["epoch"] == best["epoch"]:
            line += "   <-- selected"
        print(line)

    print(f"\nselected epoch {best['epoch']} (logP/bigram {best['logp']:.4f}, "
          f"gap {best['gap']:.4f} from the reference), on ciphertext only")

    if args.show_accuracy and any("accuracy" in r for r in rows):
        scored = [r for r in rows if r.get("accuracy") is not None]
        if scored:
            oracle = max(scored, key=lambda r: r["accuracy"])
            gap = oracle["accuracy"] - best.get("accuracy", float("nan"))
            print(f"best possible was epoch {oracle['epoch']} at "
                  f"{oracle['accuracy']:.4f}; selection cost {gap:+.4f}")
            if oracle["epoch"] == best["epoch"]:
                print("unsupervised selection found the optimum")

    out = args.out or os.path.join(run_dir, "selection.json")
    with open(out, "w") as fh:
        json.dump({"run": args.name, "criterion":
                   "bigram log-likelihood closest to held-out plaintext reference",
                   "reference_logp": reference,
                   "selected_epoch": best["epoch"],
                   "selected_logp": best["logp"],
                   "candidates": rows}, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
