import json

import numpy as np

RUNS = "results/runs"

def get(rid, field="final_val"):
    with open(f"{RUNS}/{rid}/final.json") as f:
        return json.load(f)[field]

def paired(pa, pb, n, label, field="final_val"):
    a = np.array([get(f"{pa}_s{s}", field) for s in range(1, n + 1)])
    b = np.array([get(f"{pb}_s{s}", field) for s in range(1, n + 1)])
    d = a - b
    sem = d.std(ddof=1) / np.sqrt(len(d))
    print(f"{label:46s} {a.mean():.4f} vs {b.mean():.4f} "
          f"delta={d.mean():+.4f} t={d.mean()/sem:+.2f}")

print("=== PHASE 2: FineWeb-Edu/BPE — FA: LLaMA-style 124M-class / FM: Mamba-2 SSM ===")
print("=== final val loss (nats), n=6 paired ===")
for reg, name in [("FA", "LLaMA"), ("FM", "Mamba")]:
    print(f"--- {name} ({reg}) ---")
    paired(f"final_{reg}_auto2", f"final_{reg}_muoncos", 6, f"{reg} EchoMuon vs muon+cos")
    paired(f"final_{reg}_auto2", f"final_{reg}_adamwcos", 6, f"{reg} EchoMuon vs adamw+cos")
    paired(f"final_{reg}_muoncos", f"final_{reg}_adamwcos", 6, f"{reg} muon+cos vs adamw+cos")
    lam = np.mean([get(f"final_{reg}_auto2_s{s}", "mean_lambda") for s in range(1, 7)])
    print(f"{reg} mean_lambda = {lam:.2f}")
print("--- sweep picks (lr) & sizes ---")
for reg in ["FA", "FM"]:
    for arm in ["muoncos", "auto2", "adamwcos"]:
        r = f"final_{reg}_{arm}_s1"
        print(f"{reg}/{arm}: lr={get(r, 'lr')} n_params={get(r, 'n_params')} "
              f"wall={get(r, 'wall_s'):.0f}s")
print("--- sweep grids: edge check ---")
GRIDS = {"muoncos": [5e-3, 1e-2, 2e-2], "auto2": [5e-3, 1e-2, 2e-2],
         "adamwcos": [3e-4, 6e-4, 1.2e-3]}
for reg in ["FA", "FM"]:
    for arm, grid in GRIDS.items():
        vals = []
        for lr in grid:
            try:
                vals.append((get(f"sweep_{reg}_{arm}_lr{lr:g}"), lr))
            except FileNotFoundError:
                pass
        vals.sort()
        edge = " (GRID EDGE!)" if vals and vals[0][1] in (grid[0], grid[-1]) else ""
        print(f"{reg}/{arm}: sweep best lr={vals[0][1]:g} val={vals[0][0]:.4f}{edge}")
