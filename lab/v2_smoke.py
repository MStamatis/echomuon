"""CPU smoke test for every NEW v2 code path (run before any GPU time):
scalar gate mode, shuffled+auto2, fixed lambda, ring-per-probe, vision val split,
dose-dataset prep, driver import + stage registry."""
import json
import os
import shutil
import subprocess
import sys

OUT = "/tmp/v2smoke"
shutil.rmtree(OUT, ignore_errors=True)

LM = ["--lr", "0.01", "--steps", "60", "--batch", "8", "--block", "64",
      "--n-layer", "2", "--n-head", "2", "--dim", "64", "--dataset", "shakespeare",
      "--eval-every", "30", "--eval-iters", "2", "--lr-schedule", "cosine",
      "--no-monitor", "--seed", "3", "--gate-every", "5", "--probe-every", "5"]


def run(rid, extra):
    r = subprocess.run([sys.executable, "-m", "src.train", "--out", OUT,
                        "--run-id", rid] + LM + extra)
    assert r.returncode == 0, f"{rid} FAILED"
    j = json.load(open(f"{OUT}/runs/{rid}/final.json"))
    assert j["final_val"] == j["final_val"] and j["final_val"] < 20, f"{rid} bad loss"
    return j


a = run("sc", ["--optimizer", "tcg", "--gate-mode", "scalar",
               "--auto-gate", "--auto-version", "2"])
print("scalar+auto2 OK      final_val", round(a["final_val"], 4), "lam", round(a["mean_lambda"], 3))

b = run("sh", ["--optimizer", "tcg", "--gate-mode", "shuffled",
               "--auto-gate", "--auto-version", "2"])
print("shuffled+auto2 OK    final_val", round(b["final_val"], 4), "lam", round(b["mean_lambda"], 3))

c = run("fl", ["--optimizer", "tcg", "--gate-mode", "normal", "--fixed-lambda", "0.5"])
assert c["fixed_lambda"] == 0.5 and c["mean_lambda"] is None
print("fixed-lambda OK      final_val", round(c["final_val"], 4))

d = run("rp", ["--optimizer", "tcg", "--gate-mode", "normal",
               "--auto-gate", "--auto-version", "2", "--ring-per-probe"])
probes = [json.loads(x) for x in open(f"{OUT}/runs/rp/log.jsonl") if "gate_lambda" in x]
assert probes and min(p["step"] for p in probes) == 40, \
    f"ring-per-probe: first probe at {min(p['step'] for p in probes) if probes else None}, expected 40"
print("ring-per-probe OK    first probe at step", min(p["step"] for p in probes),
      "(= 8 probe-intervals, as designed); n probes", len(probes))

# vision val split + baseline: same run with and without --val-frac must differ in
# eval metric (val vs test) but train identically (batches come from the sliced set,
# so losses differ — just assert both complete and val_frac is recorded)
V = ["--task", "vision", "--dataset", "cifar10", "--lr", "0.01", "--steps", "6",
     "--batch", "16", "--n-layer", "2", "--n-head", "2", "--dim", "64",
     "--eval-every", "6", "--eval-iters", "2", "--lr-schedule", "cosine",
     "--no-monitor", "--seed", "3", "--optimizer", "tcg", "--gate-mode", "normal",
     "--gate-every", "3"]
r = subprocess.run([sys.executable, "-m", "src.train", "--out", OUT,
                    "--run-id", "vv", "--val-frac", "0.1"] + V)
assert r.returncode == 0, "val-frac vision FAILED"
j = json.load(open(f"{OUT}/runs/vv/final.json"))
assert j["val_frac"] == 0.1 and j["final_acc"] is not None
print("vision val-frac OK   final_val", round(j["final_val"], 4), "acc", round(j["final_acc"], 4))

# dose dataset prep (uses the extracted sibling; no download)
from src.data import load_vision
tx, ty, _, _, _ = load_vision("data", "cifar10n10")
import numpy as np
meta = json.load(open("data/cifar10n10/meta.json"))
assert meta["label_noise"] == 0.10 and len(ty) == 50000
print("cifar10n10 prep OK   label_noise", meta["label_noise"])

# driver syntax + registry
r = subprocess.run([sys.executable, "run_experiments.py", "___nope___"],
                   capture_output=True, text=True)
assert "v2-queue" in r.stdout, "v2-queue missing from stage registry"
print("driver registry OK   (v2-queue present)")
print("\nALL V2 SMOKE TESTS PASS")
