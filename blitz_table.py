"""
blitz_table.py
==============

    python blitz_table.py

Builds the three blitz tables from checkpoints/bz_*/selection.json.

A  PERMUTATION-GROUP FAMILIES
   Every period-1 cipher is a single permutation applied identically at every
   position, so identity, atbash, shift, affine, keyword, composed and random
   substitution are all elements of the same group. Their key spaces span
   1 to 27! - twenty-eight orders of magnitude. If performance is flat across
   them, key space is excluded as a difficulty axis by construction rather
   than by correlation.

   'composed' is a substitution followed by a shift. Because the group is
   closed under composition, the result is a single permutation and cannot be
   harder than its parts. Any account under which composition depth predicts
   difficulty has to explain that.

B  PERIOD SWEEP
   polysub at periods 1, 2, 3, 5, 7, with an arbitrary permutation per position
   rather than Vigenere's rotations. Period 1 is the control and must match
   group A. The question is whether difficulty scales with p or whether the
   boundary is sharp at p > 1 - the latter is what a representational limit
   predicts, since a pointwise generator cannot express ANY position-dependent
   map regardless of period.

C  CORPUS SIZE
   Substitution at 100k to 6.1M characters. Tests whether the roughly
   one-in-two convergence rate is data-limited.
"""

import csv
import glob
import json
import os
import re
from collections import defaultdict

OUT = "results/tables"
THRESHOLD = 0.99

GROUP_A = ["atbash", "affine", "keyword", "composed"]
GROUP_B = ["poly1", "poly2", "poly3", "poly5", "poly7"]
GROUP_C = ["c100k", "c300k", "c1m", "c3m", "cfull"]

KEYSPACE = {
    "identity": "1", "atbash": "1", "shift": "27", "affine": "486",
    "keyword": "27!", "composed": "27!", "substitution": "27!",
    "poly1": "27!", "poly2": "(27!)^2", "poly3": "(27!)^3",
    "poly5": "(27!)^5", "poly7": "(27!)^7",
}
PERIOD = {"poly2": 2, "poly3": 3, "poly5": 5, "poly7": 7}
CHARS = {"c100k": "100k", "c300k": "300k", "c1m": "1M", "c3m": "3M",
         "cfull": "6.1M"}
SAMPLES = {"c100k": "663", "c300k": "1,975", "c1m": "6,593",
           "c3m": "19,780", "cfull": "40,436"}


def selected(path):
    d = json.load(open(path))
    m = [r for r in d["candidates"] if r["epoch"] == d["selected_epoch"]]
    c = m[0] if m else {}
    ref = d.get("reference_logp", d["selected_logp"])
    return {"epoch": d["selected_epoch"], "logp": d["selected_logp"],
            "gap": abs(d["selected_logp"] - ref),
            "acc": c.get("accuracy"), "key": c.get("key"),
            "entropy": c.get("entropy")}


def gather(prefix="bz_"):
    runs = defaultdict(list)
    for p in glob.glob(f"checkpoints/{prefix}*/selection.json"):
        name = os.path.basename(os.path.dirname(p))
        m = re.match(rf"{prefix}(.+)_s(\d+)$", name)
        if m:
            runs[m.group(1)].append(selected(p))
    # fold in the runs from earlier arrays that belong in these comparisons
    for p in glob.glob("checkpoints/subst_s*/selection.json"):
        runs["substitution"].append(selected(p))
    for p in glob.glob("checkpoints/fr_shift_s*/selection.json"):
        runs["shift"].append(selected(p))
    return runs


def summarise(rs):
    rs = [r for r in rs if r["acc"] is not None]
    if not rs:
        return None
    n = len(rs)
    solved = sum(1 for r in rs if r["acc"] >= THRESHOLD)
    pick = min(rs, key=lambda r: r["gap"])          # blind best-of-N
    accs = sorted(r["acc"] for r in rs)
    med = accs[n // 2] if n % 2 else (accs[n // 2 - 1] + accs[n // 2]) / 2
    return {"n": n, "solved": solved, "best": pick["acc"], "key": pick["key"],
            "median": med, "worst": accs[0], "epoch": pick["epoch"]}


def write(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    with open(f"{OUT}/{name}.md", "w") as fh:
        fh.write("| " + " | ".join(header) + " |\n")
        fh.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            fh.write("| " + " | ".join(str(x) for x in r) + " |\n")
    print(f"\n  " + " | ".join(f"{h:>12s}" for h in header))
    for r in rows:
        print("  " + " | ".join(f"{str(x):>12s}" for x in r))
    print(f"  -> {OUT}/{name}.csv and .md")


def main():
    runs = gather()
    if not runs:
        raise SystemExit("no checkpoints/bz_*/selection.json found")

    # ---------------------------------------------------------------- A ---
    order = ["atbash", "shift", "affine", "keyword", "composed", "substitution"]
    rows = []
    for c in order:
        s = summarise(runs.get(c, []))
        if s:
            rows.append([c, KEYSPACE.get(c, "?"), s["n"],
                         f"{s['solved']}/{s['n']}", f"{s['best']:.4f}",
                         f"{s['key']:.4f}", f"{s['median']:.4f}",
                         f"{s['worst']:.4f}"])
    if rows:
        print("\n=== A. Period-1 families: one permutation group ===")
        write("blitz_families",
              ["cipher", "key space", "seeds", "solved", "best-of-N", "key",
               "median", "worst"], rows)
        rates = [int(r[3].split("/")[0]) / int(r[3].split("/")[1]) for r in rows]
        bests = [float(r[4]) for r in rows]
        print(f"\n  solve rate spread {min(rates):.2f}-{max(rates):.2f}, "
              f"best-of-N spread {min(bests):.4f}-{max(bests):.4f}, "
              f"across key spaces from 1 to 27!")

    # ---------------------------------------------------------------- B ---
    rows = []
    for c in GROUP_B:
        s = summarise(runs.get(c, []))
        if s:
            rows.append([c, str(PERIOD.get(c, 1)), KEYSPACE.get(c, "?"),
                         f"{s['solved']}/{s['n']}", f"{s['best']:.4f}",
                         f"{s['key']:.4f}", f"{s['median']:.4f}",
                         f"{1/PERIOD.get(c,1):.4f}"])
    if rows:
        print("\n=== B. Period sweep: arbitrary permutation per position ===")
        write("blitz_period",
              ["condition", "period", "key space", "solved", "best-of-N",
               "key", "median", "1/p"], rows)
        print("\n  The 1/p column is what a pointwise generator can achieve by "
              "agreeing\n  with a single key row. Compare it to best-of-N.")

    # ---------------------------------------------------------------- C ---
    rows = []
    for c in GROUP_C:
        s = summarise(runs.get(c, []))
        if s:
            rows.append([c, CHARS.get(c, "?"), SAMPLES.get(c, "?"),
                         f"{s['solved']}/{s['n']}", f"{s['best']:.4f}",
                         f"{s['key']:.4f}", f"{s['median']:.4f}",
                         f"{s['worst']:.4f}"])
    if rows:
        print("\n=== C. Corpus size, substitution throughout ===")
        write("blitz_corpus",
              ["condition", "characters", "train samples", "solved",
               "best-of-N", "key", "median", "worst"], rows)

    # any condition not in a group
    known = set(GROUP_A + GROUP_B + GROUP_C + order)
    extra = [c for c in runs if c not in known]
    if extra:
        print(f"\n  ungrouped conditions present: {', '.join(sorted(extra))}")


if __name__ == "__main__":
    main()
