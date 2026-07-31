"""
collect.py - turn 30 completed runs into the tables and figures for Chapter 4.

    python collect.py

Writes into results/:
    ladder.csv        accuracy by cipher, three seeds each
    frontier.csv      accuracy by alphabet size, three seeds each
    ladder.png        accuracy against cipher complexity
    frontier.png      accuracy against alphabet size - the "where does it
                      break" curve
    summary.txt       everything as text, ready to paste into the chapter

Every figure reports individual seed values as well as the mean. A single run
is not evidence on this task: an earlier sweep showed within-condition variance
exceeding every between-condition difference.
"""

import glob
import json
import math
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT = "checkpoints"
OUT = "results"

LADDER = [("identity", "identity", 0.0),
          ("shift", "shift", math.log10(26)),
          ("substitution", "substitution", sum(math.log10(i) for i in range(1, 27))),
          ("vig3", "vigen\u00e8re-3", 3 * math.log10(27)),
          ("vig7", "vigen\u00e8re-7", 7 * math.log10(27))]

FRONTIER = [("a7", 7), ("a10", 10), ("a14", 14), ("a20", 20), ("a27", 27)]


def load(tag):
    """All seeds for one condition -> [(seed, acc, key, coverage, chance)]."""
    rows = []
    for f in sorted(glob.glob(os.path.join(CKPT, f"{tag}_s*", "eval_metrics.json"))):
        seed = int(re.search(r"_s(\d+)", f).group(1))
        d = json.load(open(f))
        rows.append((seed, d["character_accuracy"], d["key_recovery_accuracy"],
                     d.get("key_recovery_coverage", float("nan")),
                     d.get("chance_accuracy", float("nan"))))
    return rows


def majority_baseline(tag):
    """Frequency of the most common symbol: the accuracy of always emitting it.

    This is the honest trivial baseline. Uniform chance (1/V) understates it
    badly - a collapsed generator emitting only spaces already scores ~0.175 on
    English, which is far above 1/27.
    """
    import numpy as np
    p = os.path.join("data", tag, "dataset.npz")
    if not os.path.exists(p):
        return float("nan")
    d = np.load(p, allow_pickle=True)
    x = d["eval_plain"].ravel()
    x = x[x != 0]
    return float(np.bincount(x).max() / len(x))


def block(title, header, rows):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    out = [title, "-" * (sum(w) + 3 * len(w))]
    for r in [header] + rows:
        out.append("   ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    return "\n".join(out) + "\n"


def summarise(items, label_of, extra_of):
    """-> (text rows, csv rows, plot points)."""
    trows, crows, pts = [], [], []
    for tag, *_ in items:
        rows = load(tag)
        if not rows:
            trows.append([label_of(tag), "no runs", "", "", ""])
            continue
        accs = [a for _, a, _, _, _ in rows]
        keys = [k for _, _, k, _, _ in rows]
        mean = sum(accs) / len(accs)
        base = majority_baseline(tag)
        trows.append([label_of(tag),
                      "  ".join(f"{a:.3f}" for a in accs),
                      f"{mean:.3f}",
                      f"{max(keys) * 27:.0f}/27",
                      f"{base:.3f}"])
        for s, a, k, c, ch in rows:
            crows.append([tag, s, f"{a:.4f}", f"{k:.4f}", f"{c:.3f}", f"{ch:.4f}"])
        pts.append((extra_of(tag), accs, mean, base))
    return trows, crows, pts


def main():
    os.makedirs(OUT, exist_ok=True)
    text = []

    lab = {t: l for t, l, _ in LADDER}
    keysp = {t: k for t, _, k in LADDER}
    t1, c1, p1 = summarise(LADDER, lambda t: lab[t], lambda t: keysp[t])
    text.append(block("CIPHER COMPLEXITY LADDER (27 symbols, 3 seeds)",
                      ["cipher", "seeds", "mean", "best key", "majority base"], t1))

    size = {t: n for t, n in FRONTIER}
    t2, c2, p2 = summarise(FRONTIER, lambda t: f"{size[t]} symbols",
                           lambda t: size[t])
    text.append(block("ALPHABET-SIZE FRONTIER (substitution, 3 seeds)",
                      ["alphabet", "seeds", "mean", "best key", "majority base"], t2))

    with open(os.path.join(OUT, "ladder.csv"), "w") as f:
        f.write("cipher,seed,accuracy,key_recovery,coverage,chance\n")
        f.writelines(",".join(map(str, r)) + "\n" for r in c1)
    with open(os.path.join(OUT, "frontier.csv"), "w") as f:
        f.write("alphabet,seed,accuracy,key_recovery,coverage,chance\n")
        f.writelines(",".join(map(str, r)) + "\n" for r in c2)

    for pts, xlabel, fname, title, logx in [
        (p1, "key space (log$_{10}$)", "ladder.png",
         "Accuracy against cipher complexity", False),
        (p2, "alphabet size (symbols)", "frontier.png",
         "Accuracy against alphabet size", False)]:
        if not pts:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for x, accs, mean, base in pts:
            ax.plot([x] * len(accs), accs, "o", ms=5, alpha=0.45, color="#1f4e5f")
        xs = [x for x, _, _, _ in pts]
        ax.plot(xs, [m for _, _, m, _ in pts], "-", lw=2,
                color="#1f4e5f", label="mean of 3 seeds")
        ax.plot(xs, [b for _, _, _, b in pts], "--", lw=1.3,
                color="#b4623a", label="majority-class baseline")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("character accuracy")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150)
        text.append(f"wrote {OUT}/{fname}")

    s = "\n".join(text)
    open(os.path.join(OUT, "summary.txt"), "w").write(s)
    print(s)
    print("\nRead every mean against the majority-class baseline, not against "
          "uniform chance.\nA generator that has collapsed to the most common "
          "symbol already scores the\nbaseline while having learned nothing.")


if __name__ == "__main__":
    main()
