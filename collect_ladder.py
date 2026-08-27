"""Collect the cipher-family ladder into the table for the results chapter."""
import json, os
import numpy as np

# (dataset dir, label, key-space log10, leakage descriptor)
ROWS = [("identity", "identity", 0.0, "full"),
        ("shift", "shift", 1.4, "full"),
        ("a27", "substitution", 26.6, "full"),
        ("vig3", "vigen\u00e8re-3", 4.3, "partial"),
        ("vig7", "vigen\u00e8re-7", 10.0, "low")]

print(f"{'cipher':>14} {'keyspace':>9} {'leak':>8} {'best10':>7} "
      f"{'mean':>7} {'base':>7} {'norm':>7} {'succ':>6} {'zeros':>5}")
for d, label, ks, leak in ROWS:
    a = []
    for s in range(10):
        p = f"checkpoints/{d}_s{s}_gp1/eval_metrics.json"
        if os.path.exists(p):
            a.append(json.load(open(p))["character_accuracy"])
    if not a:
        print(f"{label:>14} {'no runs':>9}")
        continue
    a = np.array(a)
    x = np.load(f"data/{d}/dataset.npz", allow_pickle=True)["eval_plain"].ravel()
    x = x[x != 0]
    b = np.bincount(x).max() / len(x)
    print(f"{label:>14} {ks:>9.1f} {leak:>8} {a.max():>7.3f} {a.mean():>7.3f} "
          f"{b:>7.3f} {(a.max()-b)/(1-b):>7.3f} {sum(a>b)}/{len(a):<3} "
          f"{int((a==0).sum()):>5}")

print("\nThe comparison that matters is shift against substitution: identical")
print("leakage, key spaces 25 orders of magnitude apart.")
print("  shift >> substitution  -> key space governs difficulty")
print("  shift ~= substitution  -> leakage governs, and the frontier needs")
print("                            another explanation")
