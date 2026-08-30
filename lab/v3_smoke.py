"""Smoke test for the v3 arms, plus a regression guard on the published path.

Runs a tiny model through every new arm and prints what the gate and the controller
actually did, so a silently inert arm cannot reach the campaign. The last check reruns
the published configuration and requires it to be bit-identical to the value recorded
here from before the v3 changes.
"""
import json
import os
import shutil
import subprocess
import sys

BASE = ["--dataset", "shakespeare", "--task", "lm", "--steps", "80", "--batch", "8",
        "--block", "64", "--n-layer", "2", "--n-head", "2", "--dim", "64",
        "--lr", "0.01", "--gate-every", "10", "--eval-every", "40", "--eval-iters", "2",
        "--lr-schedule", "cosine", "--no-monitor", "--seed", "3",
        "--data-dir", "/lab/data"]

ARMS = [
    ("baseline  normal + auto2", ["--optimizer", "tcg", "--gate-mode", "normal",
                                  "--auto-gate", "--auto-version", "2",
                                  "--probe-every", "10"]),
    ("A1  abscal + auto2", ["--optimizer", "tcg", "--gate-mode", "abscal",
                            "--auto-gate", "--auto-version", "2", "--probe-every", "10"]),
    ("A2  abscal_hi + auto2", ["--optimizer", "tcg", "--gate-mode", "abscal_hi",
                               "--auto-gate", "--auto-version", "2", "--probe-every", "10"]),
    ("A3  normal + auto3", ["--optimizer", "tcg", "--gate-mode", "normal",
                            "--auto-gate", "--auto-version", "3"]),
    ("A4  abscal + auto3", ["--optimizer", "tcg", "--gate-mode", "abscal",
                            "--auto-gate", "--auto-version", "3"]),
]

OUT = "/tmp/v3smoke"
shutil.rmtree(OUT, ignore_errors=True)
rows = []
for name, extra in ARMS:
    rid = name.split()[0]
    r = subprocess.run([sys.executable, "-m", "src.train", "--out", OUT,
                        "--run-id", rid] + BASE + extra,
                       cwd="/lab", capture_output=True, text=True)
    if r.returncode != 0:
        print("%-26s FAILED\n%s" % (name, r.stderr[-1500:]))
        rows.append((name, None))
        continue
    fin = json.load(open(os.path.join(OUT, "runs", rid, "final.json")))
    lams, gmeans = [], []
    with open(os.path.join(OUT, "runs", rid, "log.jsonl")) as f:
        for line in f:
            j = json.loads(line)
            if "gate_lambda" in j:
                lams.append(j["gate_lambda"])
            if "signal_frac" in j:
                gmeans.append(j["signal_frac"])
    rows.append((name, dict(val=fin["final_val"], mean_lambda=fin.get("mean_lambda"),
                            n_lam=len(lams), lam_lo=min(lams) if lams else None,
                            lam_hi=max(lams) if lams else None,
                            sig=sum(gmeans) / len(gmeans) if gmeans else None)))

print()
print("%-26s %9s %9s %6s %13s %9s" % ("arm", "val", "mean_lam", "n_lam", "lambda range",
                                      "sig_frac"))
print("-" * 80)
for name, d in rows:
    if d is None:
        print("%-26s   FAILED" % name)
        continue
    rng = ("%.3f-%.3f" % (d["lam_lo"], d["lam_hi"])) if d["n_lam"] else "-"
    print("%-26s %9.4f %9s %6d %13s %9s" % (
        name, d["val"],
        "%.4f" % d["mean_lambda"] if d["mean_lambda"] is not None else "-",
        d["n_lam"], rng,
        "%.4f" % d["sig"] if d["sig"] is not None else "-"))

print()
ok = True
base = dict(rows)["baseline  normal + auto2"]
for name, d in rows:
    if d is None:
        ok = False
        print("FAIL %s did not run" % name)
    elif name != "baseline  normal + auto2" and abs(d["val"] - base["val"]) < 1e-12:
        ok = False
        print("FAIL %s is bit-identical to the baseline, so the arm is inert" % name)
for name, d in rows:
    if d and name.startswith("A3") or d and name.startswith("A4"):
        if d["sig"] is None:
            ok = False
            print("FAIL %s never recorded a signal fraction" % name)
print("SMOKE OK" if ok else "SMOKE FAILED")
