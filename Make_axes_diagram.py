"""
make_axes_diagram.py

    python make_axes_diagram.py

The "why ciphers" slide figure: two horizontal axes showing the two properties
varied independently. Not a chart - there is no data on it - just a labelled
diagram, because the point of the slide is the design, not a result.

Written as matplotlib rather than drawn by hand so the labels match the
conditions actually run, and so it can be regenerated if a condition changes.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("slides", exist_ok=True)
plt.rcParams.update({"font.size": 15})

BLUE, GOLD, DARK = "#2E5496", "#C9922B", "#333333"

fig, ax = plt.subplots(figsize=(10.5, 5.4))
ax.set_xlim(0, 10.4); ax.set_ylim(0, 6); ax.axis("off")

# ------------------------------------------------------------- axis one ---
ax.annotate("", xy=(9.4, 4.4), xytext=(0.9, 4.4),
            arrowprops=dict(arrowstyle="-|>", lw=3, color=BLUE))
ax.text(0.9, 5.15, "KEY SPACE", fontsize=18, fontweight="bold", color=BLUE)
ax.text(0.9, 4.72, "how many possible keys", fontsize=13, color=DARK)

for x, name, ks in [(1.3, "identity", "1"), (2.9, "atbash", "1"),
                    (4.4, "shift", "27"), (6.0, "affine", "486"),
                    (8.2, "substitution", "27!")]:
    ax.plot([x], [4.4], "o", color=BLUE, ms=11, zorder=3)
    ax.text(x, 4.08, name, fontsize=13.5, ha="center", va="top", color=DARK)
    ax.text(x, 3.66, ks, fontsize=13.5, ha="center", va="top",
            color=BLUE, fontweight="bold")

ax.text(5.15, 3.12, "spanning 28 orders of magnitude", fontsize=13,
        ha="center", style="italic", color=BLUE)

# ------------------------------------------------------------- axis two ---
ax.annotate("", xy=(9.8, 1.7), xytext=(0.5, 1.7),
            arrowprops=dict(arrowstyle="-|>", lw=3, color=GOLD))
ax.text(0.9, 2.45, "POSITION DEPENDENCE", fontsize=18, fontweight="bold",
        color=GOLD)
ax.text(0.9, 2.02, "does the key change along the sequence", fontsize=13,
        color=DARK)
for x, name, fam, sub in [(0.9, "period 1", "substitution", "one key"),
                          (3.0, "period 3", "Vigenère", ""),
                          (5.1, "period 7", "Vigenère", ""),
                          (7.2, "period 15", "polysub", ""),
                          (9.3, "period 31", "polysub", "31 keys")]:
    ax.plot([x], [1.7], "o", color=GOLD, ms=11, zorder=3)
    ax.text(x, 1.38, name, fontsize=13, ha="center", va="top", color=DARK)
    ax.text(x, 1.02, fam, fontsize=11.5, ha="center", va="top", color=DARK,
            style="italic")
    if sub:
        ax.text(5.15, 0.20, "one cipher per point — the two axes are varied separately",
        fontsize=13, ha="center", style="italic", color=DARK)

fig.savefig("slides/slide_axes.png", dpi=200, bbox_inches="tight")
print("wrote slides/slide_axes.png")
