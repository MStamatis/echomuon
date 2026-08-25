"""Time-to-target analysis from existing logs (no new runs).

For each cell, two fixed targets:
  T_muon  = mean final quality of the muon+cos arm  (the hard target)
  T_adamw = mean final quality of the adamw+cos arm (the easy target)
For every run we find the first wall-clock second (and step) at which the val curve
crosses the target (linear interpolation between 250-step evals). EchoMuon's gate and
probe overhead is inside its wall clock — honest seconds.
"""
import json

import numpy as np

RUNS = "results/runs"


def final(rid, field):
    with open(f"{RUNS}/{rid}/final.json") as f:
        return json.load(f)[field]


def curve(rid, key):
    pts = []
    with open(f"{RUNS}/{rid}/log.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if key in r and "t" in r:
                pts.append((r["step"], r["t"], r[key]))
    return pts


def t_to_target(pts, target, rising):
    prev = None
    for step, t, v in pts:
        if (v >= target) if rising else (v <= target):
            if prev is None or v == prev[2]:
                return t, step
            ps, pt, pv = prev
            frac = (target - pv) / (v - pv)
            return pt + frac * (t - pt), ps + frac * (step - ps)
        prev = (step, t, v)
    return None


CELLS = [
    ("FA  (LLaMA/FineWeb)", "FA", "final_val", "val_loss", False, 6),
    ("FM  (Mamba/FineWeb)", "FM", "final_val", "val_loss", False, 6),
    ("V100A (CIFAR-100)", "V100A", "final_acc", "acc", True, 8),
    ("V100P (C100 noisy)", "V100P", "final_acc", "acc", True, 8),
    ("TIA (TinyImageNet)", "TIA", "final_acc", "acc", True, 8),
    ("TIP (Tiny noisy)", "TIP", "final_acc", "acc", True, 8),
]

for name, reg, ffield, key, rising, n in CELLS:
    tgt_muon = np.mean([final(f"final_{reg}_muoncos_s{s}", ffield) for s in range(1, n + 1)])
    tgt_adamw = np.mean([final(f"final_{reg}_adamwcos_s{s}", ffield) for s in range(1, n + 1)])
    print(f"\n=== {name}  T_muon={tgt_muon:.4f}  T_adamw={tgt_adamw:.4f} ===")
    for tgt, tname in [(tgt_muon, "T_muon "), (tgt_adamw, "T_adamw")]:
        for arm in ["muoncos", "auto2", "adamwcos"]:
            times, steps, dnf = [], [], 0
            for s in range(1, n + 1):
                r = t_to_target(curve(f"final_{reg}_{arm}_s{s}", key), tgt, rising)
                if r is None:
                    dnf += 1
                else:
                    times.append(r[0])
                    steps.append(r[1])
            if times:
                print(f"  {tname} {arm:9s}: median {np.median(times):7.0f}s "
                      f"@ step {np.median(steps):5.0f}  (reached {len(times)}/{n})")
            else:
                print(f"  {tname} {arm:9s}: never reached ({dnf}/{n} DNF)")
