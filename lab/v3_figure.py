"""The difficulty-law figure: EchoMuon's margin against the baseline's error rate.

Reads paper_v2/v3_numbers.json (regenerate it with v3_numbers.py first) and writes
paper_figs/fig_difficulty.pdf. Prints every plotted number for cross-checking.
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "paper_figs"
os.makedirs(OUT, exist_ok=True)

INK, INK2, RULE = "#21272E", "#5A6470", "#E2DFD6"
ECHO, MUON, ADAMW = "#00819F", "#B26A1F", "#7A5EA8"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
})
try:
    from matplotlib import font_manager
    import glob as _glob
    otfs = _glob.glob("/xcfonts/*.otf")
    for f in otfs:
        font_manager.fontManager.addfont(f)
    if otfs:
        plt.rcParams["font.serif"] = ["XCharter"] + plt.rcParams["font.serif"]
except Exception:
    pass

R = json.load(open("paper_v2/v3_numbers.json"))
fit = R["difficulty_law"]["fit"]
cells = R["difficulty_law"]["cells"]

FAM = [("CIFAR-10", ECHO, "o"), ("CIFAR-100", MUON, "s"), ("Tiny ImageNet", ADAMW, "^")]


def family(lab):
    if lab.startswith("Tiny"):
        return "Tiny ImageNet"
    return "CIFAR-100" if "CIFAR-100" in lab else "CIFAR-10"


print("%-24s %7s %8s %7s" % ("cell", "error%", "margin", "se"))
for c in cells:
    print("%-24s %7.1f %+8.2f %7.2f" % (c["cell"], c["baseline_error"], c["margin"], c["se"]))
print("\nfit: margin = %+.4f x error %+.3f   z=%+.2f  Q=%.1f (df %d)  zero at %.1f%%"
      % (fit["slope"], fit["intercept"], fit["z"], fit["Q"], fit["df"], fit["x_at_zero"]))

fig, ax = plt.subplots(figsize=(6.3, 3.5))

xs = np.linspace(8, 72, 200)
ys = fit["intercept"] + fit["slope"] * xs
ax.plot(xs, ys, color=INK, lw=1.3, zorder=3)

# 95% band on the slope, pivoting at the weighted mean of x
w = [1.0 / c["se"] ** 2 for c in cells]
mx = sum(wi * c["baseline_error"] for wi, c in zip(w, cells)) / sum(w)
band = 1.96 * fit["se_slope"] * np.abs(xs - mx)
ax.fill_between(xs, ys - band, ys + band, color=INK, alpha=0.07, lw=0, zorder=1)

ax.axhline(0, color=RULE, lw=0.9, zorder=1)
ax.axvline(fit["x_at_zero"], color=RULE, lw=0.9, ls=(0, (3, 4)), zorder=1)

for fam, colour, marker in FAM:
    pts = [c for c in cells if family(c["cell"]) == fam]
    ax.errorbar([c["baseline_error"] for c in pts], [c["margin"] for c in pts],
                yerr=[c["se"] for c in pts], fmt=marker, ms=5.5, color=colour,
                ecolor=colour, elinewidth=1.0, capsize=2.5, lw=0, zorder=4,
                markeredgecolor="white", markeredgewidth=0.6, label=fam)

ax.annotate("no benefit\nbelow %.0f%% error" % fit["x_at_zero"],
            xy=(fit["x_at_zero"], 0), xytext=(fit["x_at_zero"] + 1.5, -0.72),
            fontsize=8, color=INK2, ha="left", va="center")

ax.text(0.975, 0.06,
        r"$\Delta = %.4f\,e - %.2f$    $z=%.1f$,  $Q=%.1f$ (df %d)"
        % (fit["slope"], -fit["intercept"], fit["z"], fit["Q"], fit["df"]),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK)

ax.set_xlabel("baseline (scheduled Muon) error rate, %")
ax.set_ylabel("EchoMuon margin, pp")
ax.set_xlim(8, 72)
ax.set_ylim(-1.0, 2.4)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", handletextpad=0.4,
          borderaxespad=0.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_difficulty.pdf"))
print("\nwrote %s/fig_difficulty.pdf" % OUT)
