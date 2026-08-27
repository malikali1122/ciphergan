"""
make_slide_figures.py
=====================

    python make_slide_figures.py

Presentation versions of the two main-result figures. Separate from
make_figures.py because slides and papers need different things: large type,
few elements, one message per figure, and no seed-level clutter.

The format matters here. The presentation is given from a laptop screen to a
table of about eight people with no projector, so anything smaller than about
14pt is unreadable and anything with more than three series is unparseable in
the ten seconds a viewer gives it.

Numbers are taken from results/tables/ and hardcoded, so this runs anywhere
matplotlib does, including a laptop with no access to the cluster.

Outputs 1600x1000 PNG at 200 dpi, which drops into a 16:9 slide without
rescaling.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = "slides"
os.makedirs(OUT, exist_ok=True)

# Type sizes roughly double the paper figures'.
plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 21, "axes.labelsize": 18,
    "legend.fontsize": 15, "xtick.labelsize": 16, "ytick.labelsize": 16,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.4, "lines.linewidth": 3.0, "lines.markersize": 11,
    "figure.dpi": 110, "savefig.bbox": "tight",
})
BLUE, GOLD, GREY, BROWN = "#2E5496", "#C9922B", "#AAAAAA", "#8A4B08"
MAJ = 0.1747


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=200)
    plt.close(fig)
    print(f"  {OUT}/{name}.png")


# --------------------------------------------------------------- slide A ---
def slide_a():
    """Main result 1: position dependence is the boundary, and it lifts.

    Three series is the maximum a slide can carry. The reference line is what
    makes the lower curve mean something - without it, 0.14 at period 7 is
    just a small number rather than exactly what a position-blind model can
    achieve.
    """
    p = np.array([1, 2, 3, 5, 7])
    without = [0.9978, 0.6478, 0.5009, 0.2451, 0.1425]
    with_pe = [0.9991, 0.9994, 0.9967, 0.9918, 0.9734]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p, 1 / p, ":", color=GREY, marker="^", lw=2.2,
            label="what a position-blind model can reach (1/p)")
    ax.plot(p, without, "--", color=GOLD, marker="s",
            label="no positional encoding")
    ax.plot(p, with_pe, "-", color=BLUE, marker="o",
            label="with positional encoding")

    ax.annotate("", xy=(7, 0.9734), xytext=(7, 0.1425),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#444444"))
    ax.text(6.80, 0.56, "0.11 → 0.98", fontsize=17, ha="right",
            va="center", color="#444444", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))

    ax.set_xlabel("cipher period  (how far the key repeats)")
    ax.set_ylabel("character accuracy")
    ax.set_xticks(p); ax.set_ylim(-0.30, 1.14)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Position dependence is the boundary — and it lifts")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.02),
              frameon=False)
    save(fig, "slide_a_position")


# --------------------------------------------------------------- slide B ---
def slide_b():
    """Main result 2: the breaking curve.

    The classical baseline is drawn as a band rather than a line. Its figure
    is key recovery and the adversarial figure is character accuracy, so a
    second line would put two different quantities on one axis; a band
    labelled with its floor says what matters - the task stays solvable - and
    says nothing it cannot support.
    """
    p = [7, 11, 15, 21, 31]
    gan = [0.9734, 0.8825, 0.7414, 0.6919, 0.5210]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhspan(0.95, 1.02, color=BROWN, alpha=0.16, zorder=0)
    ax.text(31, 0.985, "classical baseline stays above 0.95", fontsize=14,
            ha="right", va="center", color=BROWN)

    ax.plot(p, gan, "-", color=BLUE, marker="o", label="CycleGAN")
    for x, y in zip(p, gan):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=14, color=BLUE)

    ax.axhline(MAJ, color="k", lw=1.4, ls=":")
    ax.text(7, MAJ + 0.03, "trivial baseline", fontsize=14)

    ax.set_xlabel("cipher period  (key length)")
    ax.set_ylabel("character accuracy")
    ax.set_xticks(p); ax.set_ylim(-0.05, 1.12)
    ax.set_title("Where the method breaks — and what to compare against")
    save(fig, "slide_b_breaking")


if __name__ == "__main__":
    print("building slide figures:")
    slide_a()
    slide_b()
    print("\n1600x1000 at 200 dpi — drop straight into a 16:9 slide.")
