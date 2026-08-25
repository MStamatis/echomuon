"""Gate-statistic ablation report (VP = CIFAR-10 + 20% label noise, n=8 paired).

Arms: EchoMuon (auto2, agreement gate) vs four published-alternative statistics —
ablmag (magnitude / Soft-Muon-Pion style), ablcos (Magma cossim scalar),
ablcau (Cautious sign mask), ablmix (AdEMAMix fast+slow mixture) — plus muoncos.
Pre-registered (stage_gate_ablation docstring): EchoMuon > ablmag; ablcos/ablcau
between Muon and EchoMuon; ablmix ~ Muon."""
import json

import numpy as np

RUNS = "results/runs"
N = 8

def get(rid, field="final_acc"):
    with open(f"{RUNS}/{rid}/final.json") as f:
        return json.load(f)[field]

def paired(pa, pb, label, field="final_acc"):
    a = np.array([get(f"{pa}_s{s}", field) for s in range(1, N + 1)])
    b = np.array([get(f"{pb}_s{s}", field) for s in range(1, N + 1)])
    d = a - b
    sem = d.std(ddof=1) / np.sqrt(len(d))
    print(f"{label:46s} {a.mean():.4f} vs {b.mean():.4f} "
          f"delta={d.mean():+.4f} t={d.mean()/sem:+.2f}")

ARMS = ["ablmag", "ablcos", "ablcau", "ablmix"]

print("=== Gate-statistic ablations on VP (n=8 paired), accuracy ===")
print("--- each ablation vs scheduled Muon (positive = ablation better) ---")
for arm in ARMS:
    paired(f"final_VP_{arm}", "final_VP_muoncos", f"VP {arm} vs muoncos")
print("--- EchoMuon vs each ablation (positive = agreement statistic earns it) ---")
for arm in ARMS:
    paired("final_VP_auto2", f"final_VP_{arm}", f"VP echomuon vs {arm}")
print("--- reference: EchoMuon vs muoncos ---")
paired("final_VP_auto2", "final_VP_muoncos", "VP echomuon vs muoncos")
print("--- mean lambda (controller engagement) ---")
for arm in ["auto2"] + ARMS[:3]:  # ablmix has no controller
    try:
        lam = np.mean([get(f"final_VP_{arm}_s{s}", "mean_lambda") for s in range(1, N + 1)])
        print(f"VP {arm}: mean_lambda = {lam:.2f}")
    except KeyError:
        print(f"VP {arm}: mean_lambda not recorded")


def paired_fa(pa, pb, label, n=6):
    a = np.array([get(f"{pa}_s{s}", "final_val") for s in range(1, n + 1)])
    b = np.array([get(f"{pb}_s{s}", "final_val") for s in range(1, n + 1)])
    d = a - b
    sem = d.std(ddof=1) / np.sqrt(len(d))
    print(f"{label:46s} {a.mean():.4f} vs {b.mean():.4f} "
          f"delta={d.mean():+.4f} t={d.mean()/sem:+.2f}")

try:
    print("\n=== FA dissociation test (LLaMA-162M/FineWeb, n=6 paired, val loss) ===")
    print("(negative delta = first arm better)")
    paired_fa("final_FA_ablmag", "final_FA_muoncos", "FA ablmag vs muoncos")
    paired_fa("final_FA_auto2", "final_FA_ablmag", "FA echomuon vs ablmag")
    paired_fa("final_FA_auto2", "final_FA_muoncos", "FA echomuon vs muoncos (ref)")
    lam = np.mean([get(f"final_FA_ablmag_s{s}", "mean_lambda") for s in range(1, 7)])
    print(f"FA ablmag: mean_lambda = {lam:.2f}")
except FileNotFoundError:
    print("(FA ablmag runs not complete yet)")
