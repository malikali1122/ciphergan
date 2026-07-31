"""
plot_curves.py - GAN loss curves from a run's loss_log.txt

    python plot_curves.py --name idt_wgp
    python plot_curves.py --name idt_wgp st cgm --out compare.png

Why this exists
---------------
Supervisor's instruction: "GAN loss curves. When you do these things, this is
the first thing you should actually look... otherwise you're operating in the
blind."

The specific pathology to watch for is the discriminator winning outright: it
separates real from generated too easily, stops providing gradient, and the
generator has nothing to climb. On the plots that looks like D loss collapsing
towards zero while G loss climbs and never recovers.

Reference values
----------------
  LSGAN     D at equilibrium = 0.25, G at equilibrium = 0.25
            D -> 0 means the discriminator has won and the signal is dead
  WGAN-GP   loss is unbounded and may go negative; that is normal. Watch the
            SHAPE, not the sign
  cycle     chance is lambda * ln(V-1); for V=28 that is lambda * 3.296
            so lambda=10 -> 33.0 and lambda=1 -> 3.30 mean NO reconstruction

The identity cipher run is the reference curve. It is the one case where the
model can plausibly reach a perfect solution, so its curves show what "working"
looks like on this task; every later cipher should be read against it.
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADER = re.compile(r"\(epoch:\s*(\d+),\s*iters:\s*(\d+)")
PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):\s*(-?\d+\.?\d*)")
SKIP = {"epoch", "iters", "time", "data"}


def parse(path):
    """Return {loss_name: [(global_step, value), ...]}."""
    series, seen_iters, epoch_len = {}, [], 0
    with open(path) as fh:
        for line in fh:
            h = HEADER.search(line)
            if not h:
                continue
            epoch, iters = int(h.group(1)), int(h.group(2))
            epoch_len = max(epoch_len, iters)
            step = (epoch - 1) * max(epoch_len, 1) + iters
            seen_iters.append(step)
            body = line[h.end():]
            for name, val in PAIR.findall(body):
                if name in SKIP:
                    continue
                series.setdefault(name, []).append((step, float(val)))
    return series


GROUPS = [
    (["D_A", "G_A"], "A: cipher -> plaintext"),
    (["D_B", "G_B"], "B: plaintext -> cipher"),
    (["cycle_A", "cycle_B"], "cycle reconstruction"),
    (["entropy", "gp"], "diagnostics"),
]


def plot_one(name, ckpt_dir, out):
    path = os.path.join(ckpt_dir, name, "loss_log.txt")
    if not os.path.exists(path):
        raise SystemExit(f"no loss log at {path}")
    s = parse(path)
    if not s:
        raise SystemExit(
            f"{path} contains no loss lines. Train with a small --print_freq "
            "(e.g. 100); the default writes almost nothing.")

    groups = [(ks, t) for ks, t in GROUPS if any(k in s for k in ks)]
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4))
    if len(groups) == 1:
        axes = [axes]

    for ax, (keys, title) in zip(axes, groups):
        for k in keys:
            if k in s:
                xs, ys = zip(*s[k])
                ax.plot(xs, ys, lw=1.1, label=k)
        if title.startswith(("A:", "B:")):
            ax.axhline(0.25, ls="--", c="grey", lw=0.9,
                       label="LSGAN equilibrium")
        if title.startswith("cycle"):
            ax.axhline(3.296, ls="--", c="grey", lw=0.9,
                       label="chance (lambda=1)")
        ax.set_title(title)
        ax.set_xlabel("training samples seen")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(f"GAN training curves - {name}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # a compact textual summary, useful when you cannot view images over ssh
    print(f"\n{'loss':<12}{'first':>10}{'last':>10}{'min':>10}{'max':>10}")
    for k, v in s.items():
        ys = [y for _, y in v]
        print(f"{k:<12}{ys[0]:>10.3f}{ys[-1]:>10.3f}"
              f"{min(ys):>10.3f}{max(ys):>10.3f}")

    if "D_A" in s:
        last = [y for _, y in s["D_A"]][-20:]
        mean = sum(last) / len(last)
        print(f"\nD_A over the final window: {mean:.3f}")
        if mean < 0.10:
            print("  -> discriminator has won. It separates real from generated"
                  "\n     too easily and gives the generator no usable gradient."
                  "\n     This is the failure mode to report.")
        elif abs(mean - 0.25) < 0.06:
            print("  -> at LSGAN equilibrium: the two networks are balanced.")
        else:
            print("  -> neither collapsed nor at equilibrium; inspect the shape.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", nargs="+", required=True)
    p.add_argument("--checkpoints_dir", default="./checkpoints")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    for n in a.name:
        out = a.out or os.path.join(a.checkpoints_dir, n, f"curves_{n}.png")
        plot_one(n, a.checkpoints_dir, out)


if __name__ == "__main__":
    main()
