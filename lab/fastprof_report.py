import json

import numpy as np

RUNS = "results/runs"

def get(rid, field):
    with open(f"{RUNS}/{rid}/final.json") as f:
        return json.load(f)[field]

def paired(pa, pb, n, label, field):
    a = np.array([get(f"{pa}_s{s}", field) for s in range(1, n + 1)])
    b = np.array([get(f"{pb}_s{s}", field) for s in range(1, n + 1)])
    d = a - b
    sem = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / sem if sem > 0 else float("nan")
    print(f"{label:52s} {a.mean():.4f} vs {b.mean():.4f} delta={d.mean():+.4f} t={t:+.2f}")

def walls(prefix, n):
    return np.median([get(f"{prefix}_s{s}", "wall_s") for s in range(1, n + 1)])

print("=== FAST PROFILE (gate_every 100, probe_every 200) vs STANDARD (25/100) ===")
print("--- FA (LLaMA/FineWeb), n=6, final_val (lower=better) ---")
paired("final_FA_auto2f", "final_FA_auto2", 6, "fast vs standard auto2", "final_val")
paired("final_FA_auto2f", "final_FA_muoncos", 6, "fast auto2 vs muon+cos", "final_val")
print("--- TIA (Tiny-ImageNet clean), n=8, final_acc (higher=better) ---")
paired("final_TIA_auto2f", "final_TIA_auto2", 8, "fast vs standard auto2", "final_acc")
paired("final_TIA_auto2f", "final_TIA_muoncos", 8, "fast auto2 vs muon+cos", "final_acc")
print("--- wall clock (median) ---")
for reg, n in [("FA", 6), ("TIA", 8)]:
    m = walls(f"final_{reg}_muoncos", n)
    a = walls(f"final_{reg}_auto2", n)
    f = walls(f"final_{reg}_auto2f", n)
    print(f"{reg}: muon {m:.0f}s | auto2-std {a:.0f}s (+{100*(a/m-1):.0f}%) | "
          f"auto2-fast {f:.0f}s (+{100*(f/m-1):.0f}%)")
print("--- lambdas ---")
for reg, n in [("FA", 6), ("TIA", 8)]:
    lam_f = np.mean([get(f"final_{reg}_auto2f_s{s}", "mean_lambda") for s in range(1, n + 1)])
    lam_s = np.mean([get(f"final_{reg}_auto2_s{s}", "mean_lambda") for s in range(1, n + 1)])
    print(f"{reg}: lambda fast={lam_f:.2f} standard={lam_s:.2f}")
