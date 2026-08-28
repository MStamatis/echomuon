"""Bit-exactness check: the v2 code additions must leave every EXISTING mode's
training trajectory bit-identical. Runs the same tiny EchoMuon training (gate_mode
normal, auto2 controller) once with the pre-change src (mounted at /oldsrc) and once
with the current src, then compares every logged loss and the final metrics exactly.
"""
import json
import os
import shutil
import subprocess
import sys

ARGS = ["--optimizer", "tcg", "--gate-mode", "normal", "--auto-gate",
        "--auto-version", "2", "--lr", "0.01", "--steps", "60", "--batch", "8",
        "--block", "64", "--n-layer", "2", "--n-head", "2", "--dim", "64",
        "--dataset", "shakespeare", "--probe-every", "5", "--gate-every", "5",
        "--eval-every", "30", "--eval-iters", "2", "--lr-schedule", "cosine",
        "--no-monitor", "--seed", "3", "--run-id", "bit"]


def run(cwd, out):
    shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run([sys.executable, "-m", "src.train", "--out", out,
                        "--data-dir", "/lab/data"] + ARGS, cwd=cwd)
    assert r.returncode == 0, f"run in {cwd} failed"


os.makedirs("/tmp/oldlab", exist_ok=True)
shutil.rmtree("/tmp/oldlab/src", ignore_errors=True)
shutil.copytree("/oldsrc", "/tmp/oldlab/src")
run("/tmp/oldlab", "/tmp/resOLD")
run("/lab", "/tmp/resNEW")


def rows(path):
    out = []
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            j.pop("t", None)  # wall time differs; every numeric field must not
            out.append(j)
    return out


a = rows("/tmp/resOLD/runs/bit/log.jsonl")
b = rows("/tmp/resNEW/runs/bit/log.jsonl")
assert a == b, f"LOG MISMATCH: {len(a)} vs {len(b)} rows or differing values"
fa = json.load(open("/tmp/resOLD/runs/bit/final.json"))
fb = json.load(open("/tmp/resNEW/runs/bit/final.json"))
for k in ["final_val", "best_val", "mean_lambda", "final_acc"]:
    assert fa[k] == fb[k], f"final.json {k}: {fa[k]} != {fb[k]}"
print(f"BIT-EXACT: {len(a)} log rows identical; final_val {fa['final_val']:.10f} == "
      f"{fb['final_val']:.10f}; mean_lambda {fa['mean_lambda']} == {fb['mean_lambda']}")
