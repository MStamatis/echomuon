"""Generate the EchoMuon preprint figures as PDF, directly from results/runs stats.

Outputs to paper_figs/: fig1_vision.pdf, fig2_reversal.pdf, fig3_overhead.pdf,
fig4_ablation.pdf. Prints every computed number so it can be cross-checked against
the paper tables before the PDFs are accepted.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "results/runs"
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

# register XCharter (the paper's body font) if its OTFs are mounted at /xcfonts
try:
    from matplotlib import font_manager
    import glob as _glob
    otfs = _glob.glob("/xcfonts/*.otf")
    for f in otfs:
        font_manager.fontManager.addfont(f)
    if otfs:
        plt.rcParams["font.serif"] = ["XCharter"] + plt.rcParams["font.serif"]
        print(f"figures use XCharter ({len(otfs)} faces)")
except Exception as e:
    print("XCharter not registered:", e)


def M(s):
    """Typographic minus in data labels (match the axis tick glyphs)."""
    return s.replace("-", "−")


def get(rid, field):
    with open(f"{RUNS}/{rid}/final.json") as f:
        return json.load(f)[field]


def seeds_of(prefix):
    n = 1
    while os.path.exists(f"{RUNS}/{prefix}_s{n + 1}/final.json"):
        n += 1
    return n


def paired(pa, pb, field):
    n = min(seeds_of(pa), seeds_of(pb))
    a = np.array([get(f"{pa}_s{s}", field) for s in range(1, n + 1)])
    b = np.array([get(f"{pb}_s{s}", field) for s in range(1, n + 1)])
    d = a - b
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    return d.mean(), t, n


def wall(prefix):
    # median, matching fastprof_report.py (the paper's Table 3 numbers)
    n = seeds_of(prefix)
    return np.median([get(f"{prefix}_s{s}", "wall_s") for s in range(1, n + 1)])


# ---------------- Figure 1: vision margin ----------------
V_CELLS = [("VA", "C10 clean"), ("VP", "C10 noise"), ("V100A", "C100 clean"),
           ("V100P", "C100 noise"), ("TIA", "Tiny clean"), ("TIP", "Tiny noise")]
v_d, v_t = [], []
print("=== fig1: vision paired delta (acc pp) vs muoncos ===")
for reg, lab in V_CELLS:
    d, t, n = paired(f"final_{reg}_auto2", f"final_{reg}_muoncos", "final_acc")
    v_d.append(d * 100); v_t.append(t)
    print(f"  {reg:6s} {lab:12s} {d*100:+.2f}pp t={t:+.1f} n={n}")

fig, ax = plt.subplots(figsize=(6.2, 2.5))
x = np.arange(6)
ax.bar(x, v_d, width=0.62, color=ECHO, zorder=3)
for i, (d, t) in enumerate(zip(v_d, v_t)):
    ax.text(i, d + 0.06, f"+{d:.2f}", ha="center", fontsize=8.5, fontweight="bold",
            color=INK, zorder=4)
ax.set_xticks(x, [M(f"{lab}\nt {t:+.1f}") for (_, lab), t in zip(V_CELLS, v_t)], fontsize=8)
ax.set_ylim(0, 2.15)
ax.set_yticks([0, 1, 2], ["0", "+1pp", "+2pp"])
ax.axhline(0, color=INK2, lw=0.8)
for yy in (1, 2):
    ax.axhline(yy, color=RULE, lw=0.7, ls=(0, (3, 4)), zorder=1)
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_vision.pdf")
plt.close(fig)

# ---------------- Figure 2: the reversal (LM diverging) ----------------
LM_CELLS = [("P", "bytes 11M"), ("PscaleM", "bytes 38M"), ("PscaleL", "bytes 114M"),
            ("FA", "LLaMA 162M / FineWeb"), ("FM", "Mamba / FineWeb")]
lm_d, lm_t, lm_n = [], [], []
print("=== fig2: LM paired delta (mnats) vs muoncos ===")
for reg, lab in LM_CELLS:
    d, t, n = paired(f"final_{reg}_auto2", f"final_{reg}_muoncos", "final_val")
    lm_d.append(d * 1000); lm_t.append(t); lm_n.append(n)
    print(f"  {reg:8s} {lab:22s} {d*1000:+.1f} mnats t={t:+.1f} n={n}")

fig, ax = plt.subplots(figsize=(6.2, 2.3))
y = np.arange(len(LM_CELLS))[::-1]
colors = [ECHO if d < 0 else MUON for d in lm_d]
ax.barh(y, lm_d, height=0.62, color=colors, zorder=3)
for yi, d, t, (reg, _) in zip(y, lm_d, lm_t, LM_CELLS):
    if d < -3:
        ax.text(d + 0.6, yi, M(f"{d:+.1f}  t={t:+.1f}"), va="center", ha="left",
                fontsize=8.5, fontweight="bold", color="white")
    else:
        tie = " (tie)" if reg == "FM" else ""
        off = 0.5 if d >= 0 else -0.5
        ax.text(d + off, yi, M(f"{d:+.1f}{tie}"), va="center",
                ha="left" if d >= 0 else "right", fontsize=8.5, color=INK)
ax.set_yticks(y, [lab for _, lab in LM_CELLS], fontsize=8.5)
ax.axvline(0, color=INK2, lw=1.0)
ax.set_xlim(-33, 8.5)
ax.set_xlabel("Δ final val loss vs scheduled Muon (milli-nats) — left of zero is an EchoMuon win",
              fontsize=8, color=INK2)
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_reversal.pdf")
plt.close(fig)

# ---------------- Figure 3: overhead std -> fast ----------------
print("=== fig3: wall overhead vs muoncos (std profile auto2, fast auto2f) ===")
OH_CELLS = [("FA", "LLaMA 162M"), ("TIA", "Tiny ImageNet clean"), ("TIP", "Tiny ImageNet noise")]
std_oh, fast_oh = [], []
for reg, lab in OH_CELLS:
    m = wall(f"final_{reg}_muoncos")
    s = wall(f"final_{reg}_auto2") / m - 1
    f_ = wall(f"final_{reg}_auto2f") / m - 1
    std_oh.append(s * 100); fast_oh.append(f_ * 100)
    print(f"  {reg:4s} std {s*100:+.0f}%  fast {f_*100:+.0f}%")

fig, ax = plt.subplots(figsize=(6.2, 2.3))
x = np.arange(3)
ax.bar(x - 0.17, std_oh, width=0.3, facecolor="none", edgecolor=ECHO, lw=1.6, zorder=3,
       label="standard profile")
ax.bar(x + 0.17, fast_oh, width=0.3, color=ECHO, zorder=3,
       label="fast profile (recommended)")
for i, (s, f_) in enumerate(zip(std_oh, fast_oh)):
    ax.text(i - 0.17, s + 1.2, f"{s:.0f}%", ha="center", fontsize=8.5, fontweight="bold", color=INK)
    ax.text(i + 0.17, f_ + 1.2, f"{f_:.0f}%", ha="center", fontsize=8.5, fontweight="bold", color=INK)
ax.set_xticks(x, [lab for _, lab in OH_CELLS], fontsize=8.5)
ax.set_yticks([0, 25, 50], ["0", "+25%", "+50%"])
for yy in (25, 50):
    ax.axhline(yy, color=RULE, lw=0.7, ls=(0, (3, 4)), zorder=1)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_ylim(0, 56)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_overhead.pdf")
plt.close(fig)

# ---------------- Figure 4: statistic-swap ablation ----------------
print("=== fig4: statistic swap ===")
ARMS_VP = [("auto2", "agreement\n(EchoMuon)"), ("ablmag", "magnitude"),
           ("ablcos", "Magma\nscalar*"), ("ablcau", "Cautious\nmask"), ("ablmix", "AdEMAMix\nblend")]
vp_d, vp_t = [], []
for arm, lab in ARMS_VP:
    d, t, n = paired(f"final_VP_{arm}", "final_VP_muoncos", "final_acc")
    vp_d.append(d * 100); vp_t.append(t)
    print(f"  VP {arm:7s} {d*100:+.2f}pp t={t:+.1f} n={n}")
fa_d, fa_t = [], []
for arm in ["auto2", "ablmag"]:
    d, t, n = paired(f"final_FA_{arm}", "final_FA_muoncos", "final_val")
    fa_d.append(d * 1000); fa_t.append(t)
    print(f"  FA {arm:7s} {d*1000:+.1f} mnats t={t:+.1f} n={n}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 2.5), width_ratios=[2.6, 1.4])
x = np.arange(5)
cols = [ECHO] + [INK2] * 4
a1.bar(x, vp_d, width=0.6, color=cols, zorder=3)
for i, (d, t) in enumerate(zip(vp_d, vp_t)):
    va = "bottom" if d >= 0 else "top"
    off = 0.13 if d >= 0 else -0.13
    a1.text(i, d + off, M(f"{d:+.2f}\nt {t:+.1f}") if d >= 0 else M(f"{d:+.2f}  t {t:+.1f}"),
            ha="center", va=va, fontsize=7.5, color=INK)
a1.set_xticks(x, [lab for _, lab in ARMS_VP], fontsize=7.5)
a1.axhline(0, color=INK2, lw=0.9)
a1.set_ylim(-4.6, 2.6)
a1.set_title("(a) CIFAR-10, 20% noise — acc Δ vs Muon (pp)", fontsize=8, color=INK)
x2 = np.arange(2)
a2.bar(x2, fa_d, width=0.5, color=[ECHO, INK2], zorder=3)
for i, (d, t) in enumerate(zip(fa_d, fa_t)):
    va = "bottom" if d >= 0 else "top"
    off = 1.1 if d >= 0 else -1.1
    a2.text(i, d + off, M(f"{d:+.1f}\nt {t:+.1f}"), ha="center", va=va, fontsize=7.5, color=INK)
a2.set_xticks(x2, ["agreement\n(EchoMuon)", "magnitude"], fontsize=7.5)
a2.axhline(0, color=INK2, lw=0.9)
a2.set_ylim(-38, 12)
a2.set_title("(b) FineWeb LM — Δ (mnats)", fontsize=8, color=INK)
for ax_ in (a1, a2):
    ax_.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax_.tick_params(length=0)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_ablation.pdf")
plt.close(fig)

print("figures written to", OUT)
