import json

import numpy as np

RUNS = "results/runs"

def get(rid, field="final_acc"):
    with open(f"{RUNS}/{rid}/final.json") as f:
        return json.load(f)[field]

def paired(pa, pb, n, label, field="final_acc"):
    a = np.array([get(f"{pa}_s{s}", field) for s in range(1, n + 1)])
    b = np.array([get(f"{pb}_s{s}", field) for s in range(1, n + 1)])
    d = a - b
    sem = d.std(ddof=1) / np.sqrt(len(d))
    print(f"{label:44s} {a.mean():.4f} vs {b.mean():.4f} "
          f"delta={d.mean():+.4f} t={d.mean()/sem:+.2f}")

print("=== PHASE 1b: Tiny-ImageNet-200 (TIA clean / TIP 20% label noise), n=8 paired ===")
print("--- EchoMuon vs muon+cos ---")
paired("final_TIA_auto2", "final_TIA_muoncos", 8, "TIA clean acc")
paired("final_TIP_auto2", "final_TIP_muoncos", 8, "TIP noisy acc")
print("--- EchoMuon vs adamw+cos ---")
paired("final_TIA_auto2", "final_TIA_adamwcos", 8, "TIA clean acc")
paired("final_TIP_auto2", "final_TIP_adamwcos", 8, "TIP noisy acc")
print("--- loss (paired, final_val) ---")
paired("final_TIA_auto2", "final_TIA_muoncos", 8, "TIA clean loss", "final_val")
paired("final_TIP_auto2", "final_TIP_muoncos", 8, "TIP noisy loss", "final_val")
print("--- lambdas ---")
for reg in ["TIA", "TIP"]:
    lam = np.mean([get(f"final_{reg}_auto2_s{s}", "mean_lambda") for s in range(1, 9)])
    print(f"{reg} mean_lambda = {lam:.2f}")
print("--- best lrs used ---")
for reg in ["TIA", "TIP"]:
    for arm in ["muoncos", "auto2", "adamwcos"]:
        print(f"{reg}/{arm}: lr={get(f'final_{reg}_{arm}_s1', 'lr')}")
