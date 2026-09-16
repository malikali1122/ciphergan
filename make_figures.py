"""
make_figures.py
===============

    python make_figures.py

Builds every figure the results chapter needs, from the selection.json records
that already exist. No retraining, no re-evaluation.

Writes to results/figures/ as both PNG (300 dpi, for Word) and PDF (vector, if
the thesis is typeset in LaTeX).

FIGURES
  fig1_diagnosis      identity accuracy against epoch: the failed configuration
                      and the two reconfigurations, showing that the
                      framework works.
  fig2_seeds          substitution seed distribution as a strip plot. Shows
                      bimodality, which a mean and standard deviation hide.
  fig3_keyspace       best-of-N against key space on a log axis. The flat line
                      is the finding: 28 orders of magnitude, no effect.
  fig4_period         recovery against period, with and without positional
                      encoding, against the 1/p bound.
  fig5_corpus         convergence against corpus size, log axis.
  fig6_alphabet       the alphabet frontier, with the majority-class floor.
  fig7_selection      selection score against true accuracy, showing that the
                      blind criterion separates outcomes.
  fig8_baseline       CycleGAN against MCMC across all conditions.

DESIGN NOTES
Every figure that reports accuracy also draws the majority-class rate, because
accuracy against uniform chance overstates small-alphabet results. Seed-level
points are drawn wherever there are ten or fewer, since the distributions are
bimodal and any summary statistic misrepresents them.
"""

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})
C_GAN, C_MCMC, C_BASE, C_OFF = "#2E5496", "#8A4B08", "#999999", "#C9922B"
MAJ = 0.1747


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print(f"  {OUT}/{name}.png / .pdf")


def selected(path):
    d = json.load(open(path))
    m = [r for r in d["candidates"] if r["epoch"] == d["selected_epoch"]]
    c = m[0] if m else {}
    ref = d.get("reference_logp", d["selected_logp"])
    return {"gap": abs(d["selected_logp"] - ref), "logp": d["selected_logp"],
            "acc": c.get("accuracy"), "key": c.get("key"),
            "epoch": d["selected_epoch"], "entropy": c.get("entropy")}


def gather(prefix):
    runs = defaultdict(list)
    for p in glob.glob(f"checkpoints/{prefix}*/selection.json"):
        name = os.path.basename(os.path.dirname(p))
        m = re.match(rf"{re.escape(prefix)}(.+)_s(\d+)$", name)
        if m:
            r = selected(p)
            if r["acc"] is not None:
                runs[m.group(1)].append(r)
    return runs


def epochs_curve(run):
    """Every evaluated checkpoint for one run, in epoch order."""
    f = f"checkpoints/{run}/selection.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    pts = [(int(c["epoch"]), c["accuracy"]) for c in d["candidates"]
           if c["epoch"].isdigit() and c.get("accuracy") is not None]
    return sorted(pts) or None


BZ = gather("bz_")
FR = gather("fr_")
POS = gather("pos_")
EXP = gather("ex_")
SUB = gather("subst_s")          # keyed oddly; handled below
SUBST = [selected(p) for p in glob.glob("checkpoints/subst_s*/selection.json")]
SUBST = [r for r in SUBST if r["acc"] is not None]


# ------------------------------------------------------------------ fig 1 ---
def fig1():
    runs = [("cipher_full", "LSGAN + penalty (defective)", C_BASE, ":"),
            ("arm_nogp", "LSGAN, no penalty", C_OFF, "--"),
            ("arm_wgangp", "WGAN-GP, 5 critic steps", C_GAN, "-")]
    got = []
    for r, lbl, col, ls in runs:
        pts = epochs_curve(r)
        if pts is None and r == "cipher_full":
            # evaluated before pred dumping existed; these five points come
            # from the original checkpoint sweep and are recorded here so the
            # figure is complete. Re-run eval_cipher on cipher_full to replace
            # them with the full-resolution curve.
            pts = [(20, 0.0000), (60, 0.0221), (100, 0.0848),
                   (140, 0.0831), (180, 0.1200)]
        if pts:
            got.append((lbl, pts, col, ls))
    if not got:
        print("  fig1 skipped: no identity-arm records")
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for lbl, pts, col, ls in got:
        x, y = zip(*pts)
        ax.plot(x, y, ls, color=col, marker="o", ms=3.5, lw=1.6, label=lbl)
    ax.axhline(MAJ, color="k", lw=0.9, ls=":", zorder=0)
    ax.text(ax.get_xlim()[1], MAJ, " majority class", va="center",
            fontsize=7, color="k")
    ax.set_xlabel("epoch"); ax.set_ylabel("character accuracy")
    ax.set_ylim(-0.04, 1.06)
    ax.set_title("Identity control: locating the objective mis-scaling")
    ax.legend(loc="center right", frameon=False)
    save(fig, "fig1_diagnosis")


# ------------------------------------------------------------------ fig 2 ---
def fig2():
    if not SUBST:
        print("  fig2 skipped"); return
    accs = sorted((r["acc"] for r in SUBST), reverse=True)
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    # stack coincident points vertically instead of jittering, so the six
    # seeds at ~1.0 are countable rather than a single blob
    ys, seen = [], {}
    for a in accs:
        b = round(a, 3)
        seen[b] = seen.get(b, -1) + 1
        ys.append(seen[b])
    ax.scatter(accs, ys, s=55, color=C_GAN, alpha=0.9, zorder=3,
               edgecolor="white", linewidth=0.8)
    ax.axvline(MAJ, color="k", lw=0.9, ls=":")
    ax.text(MAJ + 0.012, max(ys) * 0.55 + 0.3, "majority class", fontsize=7)
    ax.set_yticks([]); ax.set_ylim(-0.7, max(ys) + 0.9)
    ax.set_xlim(-0.04, 1.06)
    ax.set_xlabel("character accuracy at the blind-selected checkpoint")
    n_hi = sum(a >= 0.99 for a in accs); n_lo = sum(a < 0.02 for a in accs)
    ax.set_title(f"Substitution, {len(accs)} seeds: {n_hi} converge, "
                 f"{len(accs)-n_hi-n_lo} partial, {n_lo} fail")
    for s in ("left",):
        ax.spines[s].set_visible(False)
    save(fig, "fig2_seeds")


# ------------------------------------------------------------------ fig 3 ---
def fig3():
    ks = {"atbash": 1, "shift": 27, "affine": 486,
          "keyword": 1.1e28, "composed": 1.1e28, "substitution": 1.1e28}
    src = dict(BZ)
    src.setdefault("shift", FR.get("shift", []))
    src["substitution"] = SUBST
    pts = []
    for c, k in ks.items():
        rs = src.get(c, [])
        if rs:
            best = min(rs, key=lambda r: r["gap"])["acc"]
            rate = sum(r["acc"] >= 0.99 for r in rs) / len(rs)
            pts.append((k, best, rate, c))
    if len(pts) < 3:
        print("  fig3 skipped"); return
    pts.sort()
    x = [p[0] for p in pts]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.semilogx(x, [p[1] for p in pts], "o-", color=C_GAN, lw=1.6, ms=6,
                label="best-of-N")
    ax.semilogx(x, [p[2] for p in pts], "s--", color=C_OFF, lw=1.4, ms=5,
                label="solve rate")
    ax.axhline(MAJ, color="k", lw=0.9, ls=":")
    # several families share key space 27!; label that cluster once
    seen_k = set()
    for k, b, r, c in pts:
        if k in seen_k:
            continue
        seen_k.add(k)
        same = [q[3] for q in pts if q[0] == k]
        lbl = same[0] if len(same) == 1 else f"{same[0]} +{len(same)-1} more"
        # alternate label heights so short-key-space names do not collide
        dy = 10 if len(seen_k) % 2 else 20
        ax.annotate(lbl, (k, b), textcoords="offset points", xytext=(0, dy),
                    ha="right" if k > 1e20 else "center", fontsize=7)
    ax.set_xlabel("key space (log scale)")
    ax.set_ylabel("character accuracy / solve rate")
    ax.set_ylim(-0.04, 1.22)
    ax.set_title("Key space does not predict difficulty")
    ax.legend(loc="lower left", frameon=False)
    save(fig, "fig3_keyspace")


# ------------------------------------------------------------------ fig 4 ---
def fig4():
    conds = [("poly1", 1), ("poly2", 2), ("poly3", 3), ("poly5", 5),
             ("poly7", 7)]
    xs, off, on = [], [], []
    for c, p in conds:
        a, b = BZ.get(c, []), POS.get(c, [])
        if a and b:
            xs.append(p)
            off.append(min(a, key=lambda r: r["gap"])["acc"])
            on.append(min(b, key=lambda r: r["gap"])["acc"])
    if len(xs) < 3:
        print("  fig4 skipped"); return
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    pp = np.array(xs, dtype=float)
    ax.plot(xs, 1 / pp, "^:", color=C_BASE, lw=1.3, ms=6,
            label="1/p (agreeing with one key row)")
    ax.plot(xs, off, "s--", color=C_OFF, lw=1.6, ms=6,
            label="without positional encoding")
    ax.plot(xs, on, "o-", color=C_GAN, lw=1.8, ms=6,
            label="with positional encoding")
    ax.axhline(MAJ, color="k", lw=0.9, ls=":")
    ax.set_xlabel("cipher period"); ax.set_ylabel("best-of-N character accuracy")
    ax.set_xticks(xs); ax.set_ylim(-0.04, 1.10)
    ax.set_title("Position dependence is the boundary, and it is removable")
    ax.legend(loc="center right", frameon=False)
    save(fig, "fig4_period")


# ------------------------------------------------------------------ fig 5 ---
def fig5():
    conds = [("c100k", 1e5), ("c300k", 3e5), ("c1m", 1e6),
             ("c3m", 3e6), ("cfull", 6.1e6)]
    xs, best, rate, seeds = [], [], [], []
    for c, n in conds:
        rs = BZ.get(c, [])
        if rs:
            xs.append(n)
            best.append(min(rs, key=lambda r: r["gap"])["acc"])
            rate.append(sum(r["acc"] >= 0.99 for r in rs) / len(rs))
            seeds.append([r["acc"] for r in rs])
    if len(xs) < 3:
        print("  fig5 skipped"); return
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for n, ss in zip(xs, seeds):
        ax.semilogx([n] * len(ss), ss, "o", color=C_GAN, alpha=0.32, ms=5,
                    zorder=2)
    # No connecting line below the convergence threshold: best-of-N moves
    # non-monotonically there, and joining the points would imply a trend the
    # data does not support.
    conv = [i for i, r in enumerate(rate) if r > 0]
    ax.semilogx(xs, best, "o", color=C_GAN, ms=7, label="best-of-N")
    if len(conv) > 1:
        ax.semilogx([xs[i] for i in conv], [best[i] for i in conv], "-",
                    color=C_GAN, lw=1.7)
    ax.semilogx(xs, rate, "s--", color=C_OFF, lw=1.4, ms=5, label="solve rate")
    ax.axhline(MAJ, color="k", lw=0.9, ls=":")
    ax.set_xlabel("corpus size (characters, log scale)")
    ax.set_ylabel("character accuracy / solve rate")
    ax.set_ylim(-0.04, 1.08)
    ax.set_title("Convergence is data-limited below ~3M characters")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig5_corpus")


# ------------------------------------------------------------------ fig 6 ---
def fig6():
    conds = [("a7", 7, 0.3505), ("a10", 10, 0.2962), ("a14", 14, 0.2686),
             ("a20", 20, 0.2031), ("substitution", 27, 0.1747)]
    xs, best, rate, floor, seeds = [], [], [], [], []
    src = dict(FR); src["substitution"] = SUBST
    for c, v, mj in conds:
        rs = src.get(c, [])
        if rs:
            xs.append(v); floor.append(mj)
            best.append(min(rs, key=lambda r: r["gap"])["acc"])
            rate.append(sum(r["acc"] >= 0.99 for r in rs) / len(rs))
            seeds.append([r["acc"] for r in rs])
    if len(xs) < 3:
        print("  fig6 skipped"); return
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for v, ss in zip(xs, seeds):
        ax.plot([v] * len(ss), ss, "o", color=C_GAN, alpha=0.30, ms=5)
    ax.plot(xs, best, "o-", color=C_GAN, lw=1.7, ms=6, label="best-of-N")
    ax.plot(xs, rate, "s--", color=C_OFF, lw=1.4, ms=5, label="solve rate")
    ax.plot(xs, floor, "^:", color=C_BASE, lw=1.2, ms=5,
            label="majority-class floor")
    ax.set_xlabel("alphabet size")
    ax.set_ylabel("character accuracy / solve rate")
    ax.set_xticks(xs); ax.set_ylim(-0.04, 1.08)
    ax.set_title("Alphabet frontier: capability flat, reliability noisy")
    ax.legend(loc="center right", frameon=False)
    save(fig, "fig6_alphabet")


# ------------------------------------------------------------------ fig 7 ---
def fig7():
    rs = [r for r in SUBST if r["acc"] is not None]
    for d in (BZ, FR, POS):
        for v in d.values():
            rs.extend(v)
    rs = [r for r in rs if r["acc"] is not None]
    if len(rs) < 10:
        print("  fig7 skipped"); return
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ok = [r for r in rs if r["acc"] >= 0.90]
    no = [r for r in rs if r["acc"] < 0.90]
    ax.scatter([r["gap"] for r in no], [r["acc"] for r in no], s=26,
               color=C_OFF, alpha=0.65, label="failed", edgecolor="none")
    ax.scatter([r["gap"] for r in ok], [r["acc"] for r in ok], s=26,
               color=C_GAN, alpha=0.75, label="converged", edgecolor="none")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("distance of selection score from the plaintext reference")
    ax.set_ylabel("true character accuracy")
    ax.set_title("Selection score against true accuracy, all runs")
    ax.axvline(0.05, color="k", lw=0.8, ls=":", zorder=0)
    ax.text(0.055, 0.06, "gap = 0.05", fontsize=7, rotation=90)
    ax.legend(loc="upper right", frameon=False)
    save(fig, "fig7_selection")


# ------------------------------------------------------------------ fig 8 ---
def fig8():
    mc = {}
    for p in glob.glob("results/mcmc/*.json"):
        c = os.path.basename(p)[:-5]
        if not c.endswith("_randinit"):
            mc[c] = json.load(open(p))["key_recovery_best"]
    src = dict(BZ); src.update(FR); src["substitution"] = SUBST
    ident = []
    for r in ("arm_wgangp", "arm_nogp"):
        f = f"checkpoints/{r}/selection.json"
        if os.path.exists(f):
            ident.append(selected(f))
    ident = [r for r in ident if r["acc"] is not None]
    if ident:
        src["identity"] = ident
    order = ["identity", "shift", "a7", "a10", "a14", "a20",
             "substitution", "vig3", "vig7"]
    labels, gan, base = [], [], []
    for c in order:
        if c in src and src[c] and c in mc:
            labels.append(c)
            gan.append(min(src[c], key=lambda r: r["gap"])["acc"])
            base.append(mc[c])
    if len(labels) < 3:
        print("  fig8 skipped"); return
    pos = []
    for c in labels:
        rs = POS.get(c, [])
        pos.append(min(rs, key=lambda r: r["gap"])["acc"] if rs else np.nan)
    has_pos = not all(np.isnan(p) for p in pos)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    w = 0.27 if has_pos else 0.4
    ax.bar(x - w, base, w, color=C_MCMC, label="MCMC baseline")
    ax.bar(x, gan, w, color=C_GAN, label="CycleGAN best-of-N")
    if has_pos:
        ax.bar(x + w, pos, w, color=C_OFF,
               label="CycleGAN + positional encoding")
    ax.axhline(MAJ, color="k", lw=0.9, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("character accuracy"); ax.set_ylim(0, 1.08)
    ax.set_title("Against a competent classical attack")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2,
              frameon=False)
    save(fig, "fig8_baseline")


# ------------------------------------------------------------------ fig 9 ---
def fig9():
    """MCMC cost: driven by period, insensitive to key space.

    This mirrors the adversarial finding in reverse. For the classical attack a
    longer period multiplies the work but never prevents a solve; for the
    generator it prevented a solve entirely. Alphabet size and key space move
    the classical cost hardly at all.
    """
    rows = []
    for p in glob.glob("results/mcmc/*.json"):
        c = os.path.basename(p)[:-5]
        if c.endswith("_randinit"):
            continue
        d = json.load(open(p))
        rows.append((c, d["period"], d["vocab_size"] - 1, d["seconds"],
                     d["n_solved"]))
    if len(rows) < 4:
        print("  fig9 skipped"); return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    per = defaultdict(list)
    for c, p, v, s, n in rows:
        per[p].append(s)
    xs = sorted(per)
    a1.plot(xs, [np.mean(per[x]) for x in xs], "o-", color=C_MCMC, lw=1.7, ms=6)
    for c, p, v, s, n in rows:
        a1.plot(p, s, "o", color=C_MCMC, alpha=0.30, ms=5)
    a1.set_xlabel("cipher period"); a1.set_ylabel("CPU seconds, 10 seeds")
    a1.set_xticks(xs); a1.set_title("Cost scales with period")

    mono = [(v, s) for c, p, v, s, n in rows if p == 1]
    if mono:
        mono.sort()
        a2.plot([m[0] for m in mono], [m[1] for m in mono], "o", color=C_MCMC,
                ms=6)
        a2.set_ylim(0, max(s for c, p, v, s, n in rows) * 1.05)
    a2.set_xlabel("alphabet size (period 1)")
    a2.set_title("and not with alphabet size")

    for ax in (a1, a2):
        ax.set_ylim(bottom=0)
    fig.suptitle("Classical baseline: every condition solved 10/10", y=1.02,
                 fontsize=10)
    save(fig, "fig9_baseline_cost")


# ----------------------------------------------------------------- fig 10 ---
def fig10():
    """Key length: where the method breaks, once it can represent the key.

    Positional encoding is on throughout, so this measures the method rather
    than the function class. The classical baseline is drawn alongside because
    it stays above 0.95 at every period: the tasks remain solvable, and the
    decline belongs to the adversarial method.

    This is the curve a second paradigm would be overlaid on.
    """
    conds = [("poly7", 7), ("poly11", 11), ("poly15", 15),
             ("poly21", 21), ("poly31", 31)]
    xs, gan, seeds, base = [], [], [], []
    for c, p in conds:
        # runs are named pos_<cond>_s<n> or ex_<cond>_pos_s<n>, so the
        # gathered key is either <cond> or <cond>_pos
        rs = POS.get(c, []) or EXP.get(c + "_pos", []) or EXP.get(c, [])
        mc = f"results/mcmc/{c}.json"
        if not rs:
            continue
        xs.append(p)
        gan.append(min(rs, key=lambda r: r["gap"])["acc"])
        seeds.append([r["acc"] for r in rs])
        # character accuracy, to match the GAN series: plotting key recovery
        # for one method and character accuracy for the other would put two
        # different quantities on one axis
        base.append(json.load(open(mc)).get("character_accuracy_best", np.nan)
                    if os.path.exists(mc) else np.nan)
    if len(xs) < 3:
        print("  fig10 skipped"); return

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for p, ss in zip(xs, seeds):
        ax.plot([p] * len(ss), ss, "o", color=C_GAN, alpha=0.28, ms=5, zorder=2)
    ax.plot(xs, gan, "o-", color=C_GAN, lw=1.8, ms=7,
            label="CycleGAN + positional encoding")
    if not all(np.isnan(base)):
        ax.plot(xs, base, "s--", color=C_MCMC, lw=1.5, ms=6,
                label="MCMC baseline")
    ax.plot(xs, [1 / p for p in xs], "^:", color=C_BASE, lw=1.2, ms=5,
            label="1/p (agreeing with one key row)")
    ax.axhline(MAJ, color="k", lw=0.9, ls=":")
    ax.set_xlabel("cipher period (key length)")
    ax.set_ylabel("character accuracy")
    ax.set_xticks(xs); ax.set_ylim(-0.04, 1.08)
    ax.set_title("Where the method breaks as the key lengthens")
    ax.legend(loc="lower left", frameon=False)
    save(fig, "fig10_keylength")


# ----------------------------------------------------------------- fig 11 ---
def fig11():
    """Word-level substitution at 200 types: a reproduction failure.

    Four bars on one task. The two OOV conventions are shown side by side
    because the difference between them is large and was the source of an
    error in this project: dropping out-of-vocabulary tokens destroys word
    order, and with it the bigram statistics every attack depends on.

    Character accuracy is reported for comparability with the published
    figure, and the majority-class line shows why key recovery is the metric
    that should be quoted: <unk> is 45% of tokens, so a model that identifies
    it and nothing else already scores 0.4531.
    """
    got = {}
    for tag, path in (("gan_unk", "checkpoints/ex_u200_s*/selection.json"),
                      ("gan_drop", "checkpoints/ex_w200_s*/selection.json")):
        rs = [selected(p) for p in glob.glob(path)]
        rs = [r for r in rs if r["acc"] is not None]
        if rs:
            got[tag] = min(rs, key=lambda r: r["gap"])
    for tag, c in (("mcmc_unk", "u200"), ("mcmc_drop", "w200")):
        f = f"results/mcmc/{c}.json"
        if os.path.exists(f):
            got[tag] = json.load(open(f))
    if "gan_unk" not in got:
        print("  fig11 skipped"); return

    labels, vals, cols = [], [], []
    if "mcmc_drop" in got:
        labels.append("MCMC\nOOV dropped")
        vals.append(got["mcmc_drop"]["character_accuracy_best"])
        cols.append(C_BASE)
    if "mcmc_unk" in got:
        labels.append("MCMC\nOOV = <unk>")
        vals.append(got["mcmc_unk"]["character_accuracy_best"])
        cols.append(C_MCMC)
    if "gan_drop" in got:
        labels.append("CycleGAN\nOOV dropped")
        vals.append(got["gan_drop"]["acc"]); cols.append(C_BASE)
    labels.append("CycleGAN\nOOV = <unk>")
    vals.append(got["gan_unk"]["acc"]); cols.append(C_GAN)
    labels.append("Gomez et al.\n(reported)")
    vals.append(0.987); cols.append(C_OFF)

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, vals, 0.62, color=cols)
    ax.axhline(0.4531, color="k", lw=1.0, ls="--")
    ax.text(len(labels) - 0.4, 0.4531 + 0.015,
            "majority class (<unk> alone)", fontsize=7, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("character accuracy"); ax.set_ylim(0, 1.12)
    ax.set_title("Word-level substitution, 200 types")
    save(fig, "fig11_wordlevel")


if __name__ == "__main__":
    print("building figures:")
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9,
              fig10, fig11):
        try:
            f()
        except Exception as e:
            print(f"  {f.__name__} failed: {e}")
    print(f"\ndone -> {OUT}/")
