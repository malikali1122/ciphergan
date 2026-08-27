"""
frontier_table.py
=================

Collapse the per-seed frontier runs into the table and the plot the results
chapter needs.

    python frontier_table.py

Reads checkpoints/fr_<cond>_s<seed>/selection.json, plus the substitution runs
(checkpoints/subst_s*) as the 27-symbol point, and the MCMC results in
results/mcmc/ as the baseline column.

Writes results/tables/frontier.csv / .md and, if matplotlib is available,
results/tables/frontier.png.

TWO NUMBERS PER CONDITION, and both belong in the chapter:

  solve rate    how many seeds reached the threshold. This is the reliability
                finding, and it is where the bimodality Gomez et al. report
                shows up.
  best-of-N     the accuracy of the single seed whose decryption best matches
                the plaintext reference. Chosen without labels, and the same
                protocol the MCMC baseline uses for its restarts, so the two
                methods are compared on equal terms.

Quoting only the first understates the method; quoting only the second hides
that most seeds failed.
"""

import glob
import json
import os
import re
from collections import defaultdict

OUT = "results/tables"
THRESHOLD = 0.99

# how conditions order along the frontier, and their usable alphabet size
ORDER = ["identity", "shift", "a7", "a10", "a14", "a20", "substitution",
         "vig3", "vig7"]
ALPHA = {"a7": 7, "a10": 10, "a14": 14, "a20": 20, "substitution": 27,
         "identity": 27, "shift": 27, "vig3": 27, "vig7": 27}
PERIOD = {"vig3": 3, "vig7": 7}


def selected(path):
    d = json.load(open(path))
    m = [r for r in d["candidates"] if r["epoch"] == d["selected_epoch"]]
    c = m[0] if m else {}
    return {"epoch": d["selected_epoch"], "logp": d["selected_logp"],
            "gap": abs(d["selected_logp"] - d.get("reference_logp",
                                                  d["selected_logp"])),
            "acc": c.get("accuracy"), "key": c.get("key")}


def gather():
    runs = defaultdict(list)
    for p in glob.glob("checkpoints/fr_*/selection.json"):
        m = re.match(r"fr_(.+)_s(\d+)$", os.path.basename(os.path.dirname(p)))
        if m:
            runs[m.group(1)].append(selected(p))
    for p in glob.glob("checkpoints/subst_s*/selection.json"):
        runs["substitution"].append(selected(p))
    return runs


def baseline():
    out = {}
    for p in glob.glob("results/mcmc/*.json"):
        c = os.path.basename(p)[:-5]
        if c.endswith("_randinit"):
            continue
        d = json.load(open(p))
        out[c] = (d["n_solved"], d["key_recovery_best"], d["seconds"])
    return out


def main():
    runs, base = gather(), baseline()
    if not runs:
        raise SystemExit("no selection.json found - run sweep_frontier.sh first")

    conds = [c for c in ORDER if c in runs] + \
            sorted(c for c in runs if c not in ORDER)
    rows = []
    for c in conds:
        rs = [r for r in runs[c] if r["acc"] is not None]
        if not rs:
            continue
        n = len(rs)
        solved = sum(1 for r in rs if r["acc"] >= THRESHOLD)
        # best-of-N chosen by the blind criterion, not by accuracy
        pick = min(rs, key=lambda r: r["gap"])
        accs = sorted(r["acc"] for r in rs)
        med = accs[n // 2] if n % 2 else (accs[n // 2 - 1] + accs[n // 2]) / 2
        b = base.get(c)
        rows.append([
            c, PERIOD.get(c, 1), ALPHA.get(c, "?"), n,
            f"{solved}/{n}", f"{pick['acc']:.4f}", f"{pick['key']:.4f}",
            f"{med:.4f}", f"{accs[0]:.4f}",
            f"{b[0]}/10" if b else "-", f"{b[1]:.4f}" if b else "-",
        ])

    hdr = ["condition", "period", "alphabet", "seeds", "GAN solved",
           "GAN best-of-N", "key", "median", "worst",
           "MCMC solved", "MCMC key"]
    os.makedirs(OUT, exist_ok=True)
    import csv
    with open(f"{OUT}/frontier.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)
    with open(f"{OUT}/frontier.md", "w") as fh:
        fh.write("| " + " | ".join(hdr) + " |\n")
        fh.write("|" + "|".join(["---"] * len(hdr)) + "|\n")
        for r in rows:
            fh.write("| " + " | ".join(str(x) for x in r) + " |\n")

    print("\n" + " | ".join(f"{h:>13s}" for h in hdr))
    for r in rows:
        print(" | ".join(f"{str(x):>13s}" for x in r))
    print(f"\nwrote {OUT}/frontier.csv and .md")
    print(f"threshold for 'solved' is character accuracy >= {THRESHOLD}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        alpha_rows = [r for r in rows if r[1] == 1 and isinstance(r[2], int)]
        if len(alpha_rows) >= 2:
            alpha_rows.sort(key=lambda r: r[2])
            xs = [r[2] for r in alpha_rows]
            best = [float(r[5]) for r in alpha_rows]
            rate = [int(r[4].split("/")[0]) / int(r[4].split("/")[1])
                    for r in alpha_rows]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(xs, best, "o-", label="CycleGAN, best-of-N")
            ax.plot(xs, rate, "s--", label="CycleGAN, solve rate")
            ax.axhline(1.0, color="k", lw=1, ls=":",
                       label="MCMC baseline (all conditions)")
            ax.set_xlabel("alphabet size")
            ax.set_ylabel("character accuracy / solve rate")
            ax.set_ylim(-0.05, 1.08)
            ax.legend(fontsize=8)
            ax.set_title("Substitution frontier by alphabet size")
            fig.tight_layout()
            fig.savefig(f"{OUT}/frontier.png", dpi=150)
            print(f"wrote {OUT}/frontier.png")
    except ImportError:
        print("matplotlib unavailable, skipping the plot")


if __name__ == "__main__":
    main()
