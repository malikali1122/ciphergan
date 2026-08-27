"""
pos_table.py
============

    python pos_table.py

Pairs every positional-encoding run (pos_*) with its matched control (bz_* or
fr_*) and reports both against the 1/p bound.

TWO THRESHOLDS, DELIBERATELY
Reporting only "solved at 0.99" is misleading here. Period-7 with positional
encoding puts all five seeds between 0.93 and 0.98, which is the most reliable
result in the study, yet scores 0/5 at a 0.99 threshold. A 0.90 column is
reported alongside so that a real effect is not hidden by a cutoff chosen for
the monoalphabetic conditions.

THE CONTROL MATTERS
poly1 has no position dependence. Positional encoding should therefore change
nothing there. If it degrades, the encoding has a cost on position-independent
ciphers - the generator can condition on a feature that carries no signal - and
that cost has to be reported alongside the benefit.
"""

import csv
import glob
import json
import os
import re
from collections import defaultdict

OUT = "results/tables"
PERIOD = {"poly1": 1, "poly2": 2, "poly3": 3, "poly5": 5, "poly7": 7,
          "vig3": 3, "vig7": 7}


def selected(path):
    d = json.load(open(path))
    m = [r for r in d["candidates"] if r["epoch"] == d["selected_epoch"]]
    c = m[0] if m else {}
    ref = d.get("reference_logp", d["selected_logp"])
    return {"gap": abs(d["selected_logp"] - ref), "acc": c.get("accuracy"),
            "key": c.get("key"), "epoch": d["selected_epoch"]}


def gather(prefix):
    runs = defaultdict(list)
    for p in glob.glob(f"checkpoints/{prefix}*/selection.json"):
        name = os.path.basename(os.path.dirname(p))
        m = re.match(rf"{prefix}(.+)_s(\d+)$", name)
        if m:
            r = selected(p)
            if r["acc"] is not None:
                runs[m.group(1)].append(r)
    return runs


def stats(rs):
    if not rs:
        return None
    n = len(rs)
    accs = sorted(r["acc"] for r in rs)
    pick = min(rs, key=lambda r: r["gap"])      # blind best-of-N
    med = accs[n // 2] if n % 2 else (accs[n // 2 - 1] + accs[n // 2]) / 2
    return {"n": n, "best": pick["acc"], "key": pick["key"],
            "s99": sum(a >= 0.99 for a in accs),
            "s90": sum(a >= 0.90 for a in accs),
            "median": med, "worst": accs[0]}


def main():
    on = gather("pos_")
    off = gather("bz_")
    for k, v in gather("fr_").items():          # vig3, vig7 live here
        off.setdefault(k, []).extend(v)
    if not on:
        raise SystemExit("no checkpoints/pos_*/selection.json found")

    order = ["poly1", "poly2", "poly3", "poly5", "poly7", "vig3", "vig7"]
    hdr = ["condition", "period", "1/p", "best OFF", "best ON", "gain",
           "ON >=0.99", "ON >=0.90", "ON median", "ON worst"]
    rows = []
    for c in order:
        a, b = stats(off.get(c, [])), stats(on.get(c, []))
        if not b:
            continue
        p = PERIOD.get(c, 1)
        rows.append([
            c, str(p), "—" if p == 1 else f"{1/p:.4f}",
            f"{a['best']:.4f}" if a else "—",
            f"{b['best']:.4f}",
            f"{b['best'] - a['best']:+.4f}" if a else "—",
            f"{b['s99']}/{b['n']}", f"{b['s90']}/{b['n']}",
            f"{b['median']:.4f}", f"{b['worst']:.4f}"])

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/positional.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)
    with open(f"{OUT}/positional.md", "w") as fh:
        fh.write("| " + " | ".join(hdr) + " |\n")
        fh.write("|" + "|".join(["---"] * len(hdr)) + "|\n")
        for r in rows:
            fh.write("| " + " | ".join(str(x) for x in r) + " |\n")

    print("\n" + " | ".join(f"{h:>10s}" for h in hdr))
    for r in rows:
        print(" | ".join(f"{str(x):>10s}" for x in r))
    print(f"\nwrote {OUT}/positional.csv and .md")

    # per-seed detail, since the distributions are bimodal and a summary hides that
    print("\nper-seed selected accuracy, positional encoding on:")
    for c in order:
        if c in on:
            accs = sorted((r["acc"] for r in on[c]), reverse=True)
            print(f"  {c:7s} " + "  ".join(f"{a:.4f}" for a in accs))

    ctrl = stats(on.get("poly1", []))
    ctrl_off = stats(off.get("poly1", []))
    if ctrl and ctrl_off:
        print(f"\ncontrol check (poly1 has no position dependence):")
        print(f"  without encoding  {ctrl_off['s99']}/{ctrl_off['n']} at 0.99, "
              f"median {ctrl_off['median']:.4f}")
        print(f"  with encoding     {ctrl['s99']}/{ctrl['n']} at 0.99, "
              f"median {ctrl['median']:.4f}")
        if ctrl["median"] < ctrl_off["median"] - 0.01:
            print("  -> the encoding COSTS something where position carries no "
                  "signal.\n     Report this alongside the benefit; it is a "
                  "trade-off, not a free win.")


if __name__ == "__main__":
    main()
