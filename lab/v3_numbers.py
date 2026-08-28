"""Every number the v3 paper needs, recomputed from results/runs/*/final.json.

Self-contained (no imports from the v2 scripts) so it can ship as the single
provenance file for the rewritten paper. Writes paper_v2/v3_numbers.json and
prints a readable report.

Run:  python v3_numbers.py
"""
import json
import math
import os

RUNS = os.path.join("results", "runs")
OUT = "paper_v2"
os.makedirs(OUT, exist_ok=True)

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 14: 2.145, 15: 2.131}


# ---------------------------------------------------------------- primitives
def load(run):
    p = os.path.join(RUNS, run, "final.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def seeds_of(prefix):
    out = {}
    for d in sorted(os.listdir(RUNS)):
        if d.startswith(prefix + "_s"):
            tail = d[len(prefix) + 2:]
            if tail.isdigit():
                j = load(d)
                if j is not None:
                    out[int(tail)] = j
    return out


def _betacf(a, b, x):
    tiny, eps = 1e-30, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def paired2(pa, pb, metric="final_acc", scale=100.0):
    """Paired stats of prefix_a - prefix_b over shared seeds (full run prefixes)."""
    A, B = seeds_of(pa), seeds_of(pb)
    common = sorted(set(A) & set(B))
    if len(common) < 2:
        return None
    va = [A[s][metric] * scale for s in common if A[s].get(metric) is not None]
    vb = [B[s][metric] * scale for s in common if B[s].get(metric) is not None]
    if len(va) != len(common) or len(vb) != len(common):
        return None
    d = [x - y for x, y in zip(va, vb)]
    n = len(d)
    md = sum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    t = md / se if se > 0 else float("inf")
    tc = T975.get(n - 1, 2.2)
    return dict(n=n, mean_a=sum(va) / n, mean_b=sum(vb) / n, delta=md, se=se,
                t=t, p=t_two_sided_p(t, n - 1),
                ci=[md - tc * se, md + tc * se],
                lr_a=A[common[0]].get("lr"), lr_b=B[common[0]].get("lr"))


def grid(prefix):
    """[(lr, final_val)] for every completed sweep run under `prefix`."""
    r = []
    for d in os.listdir(RUNS):
        if not d.startswith(prefix):
            continue
        try:
            lr = float(d[len(prefix):])
        except ValueError:
            continue
        j = load(d)
        if j is not None:
            r.append((lr, j["final_val"]))
    return sorted(r)


def pick(prefix):
    """(best_lr, gap_over_runner_up, grid_lrs, edge_status) or None."""
    g = grid(prefix)
    if len(g) < 2:
        return None
    s = sorted(g, key=lambda x: x[1])
    lrs = [x[0] for x in g]
    edge = ("bottom" if s[0][0] == lrs[0] else
            "top" if s[0][0] == lrs[-1] else "interior")
    return dict(lr=s[0][0], gap=s[1][1] - s[0][1], grid=lrs, edge=edge)


def pooled(cells):
    """Inverse-variance pooled effect + Cochran Q over (delta, se) pairs."""
    w = [1.0 / c["se"] ** 2 for c in cells]
    W = sum(w)
    m = sum(wi * c["delta"] for wi, c in zip(w, cells)) / W
    se = math.sqrt(1.0 / W)
    Q = sum(wi * (c["delta"] - m) ** 2 for wi, c in zip(w, cells))
    return dict(delta=m, se=se, t=m / se, ci=[m - 1.96 * se, m + 1.96 * se],
                Q=Q, df=len(cells) - 1, k=len(cells))


def wls(points):
    """Weighted least squares y ~ a + b*x over (x, y, se)."""
    w = [1.0 / p[2] ** 2 for p in points]
    W = sum(w)
    mx = sum(wi * p[0] for wi, p in zip(w, points)) / W
    my = sum(wi * p[1] for wi, p in zip(w, points)) / W
    Sxx = sum(wi * (p[0] - mx) ** 2 for wi, p in zip(w, points))
    b = sum(wi * (p[0] - mx) * (p[1] - my) for wi, p in zip(w, points)) / Sxx
    a = my - b * mx
    se_b = math.sqrt(1.0 / Sxx)
    Q = sum(wi * (p[1] - (a + b * p[0])) ** 2 for wi, p in zip(w, points))
    return dict(slope=b, intercept=a, se_slope=se_b, z=b / se_b,
                ci=[b - 1.96 * se_b, b + 1.96 * se_b], Q=Q, df=len(points) - 2,
                x_at_zero=-a / b if b else None, k=len(points))


# ---------------------------------------------------------------- the numbers
R = {}

# (1) the 2x learning-rate rule -----------------------------------------------
R["lr_rule"] = []
for lab, mp, ep in [
        ("byte-LM 38M", "final_PscaleM_muoncos", "final_PscaleM_auto2"),
        ("byte-LM 114M", "final_PscaleL_muoncos", "final_PscaleL_auto2"),
        ("CIFAR-10 24k", "final_B10_muoncos", "final_B10_echomuonf"),
        ("CIFAR-100 clean", "final_V100Av_muoncos", "final_V100Am_auto2"),
        ("FineWeb 3x", "final_FA3xw_muoncos", "final_FA3x_auto2")]:
    m, e = seeds_of(mp), seeds_of(ep)
    if m and e:
        lm, le = m[sorted(m)[0]]["lr"], e[sorted(e)[0]]["lr"]
        R["lr_rule"].append(dict(cell=lab, muon_lr=lm, echo_lr=le,
                                 ratio=le / lm if lm else None))

# (2) grid-edge audit of the shipped vision cells ------------------------------
R["grid_audit"] = []
for cell in ["VA", "VP", "V100A", "V100P", "TIA", "TIP"]:
    for arm in ["muoncos", "auto2"]:
        orig = pick(f"sweep_{cell}_{arm}_lr")
        val = pick(f"sweepv_{cell}_{arm}_lr")
        multi = pick(f"sweepw_{cell}_{arm}_lr")
        R["grid_audit"].append(dict(
            cell=cell, arm=arm,
            shipped=orig, val_selected=val, split_recheck=multi))

# (3) vision ledger: shipped vs corrected --------------------------------------
LEDGER = [
    ("CIFAR-10 clean", "VA", "final_VAv_auto2", "final_VAv_muoncos"),
    ("CIFAR-10 20% noise", "VP", "final_VPv_auto2", "final_VPv_muoncos"),
    ("CIFAR-100 clean", "V100A", "final_V100Am_auto2", "final_V100Av_muoncos"),
    ("CIFAR-100 20% noise", "V100P", "final_V100Pv_auto2", "final_V100Pv_muoncos"),
    ("Tiny ImageNet clean", "TIA", "final_TIA_auto2", "final_TIA_muoncos"),
    ("Tiny ImageNet 20% noise", "TIP", "final_TIP_auto2", "final_TIP_muoncos"),
]
R["vision_ledger"] = []
for lab, cell, ea, ma in LEDGER:
    shipped = paired2(f"final_{cell}_auto2", f"final_{cell}_muoncos")
    corrected = paired2(ea, ma)
    R["vision_ledger"].append(dict(cell=lab, shipped=shipped, corrected=corrected,
                                   retained=(corrected["delta"] / shipped["delta"]
                                             if shipped and corrected and shipped["delta"]
                                             else None)))

# (4) pooled families at the corrected lr --------------------------------------
cif = [r["corrected"] for r in R["vision_ledger"]
       if r["corrected"] and "CIFAR" in r["cell"]]
tin = [r["corrected"] for r in R["vision_ledger"]
       if r["corrected"] and "Tiny" in r["cell"]]
R["pooled"] = dict(cifar=pooled(cif) if len(cif) > 1 else None,
                   tinyimagenet=pooled(tin) if len(tin) > 1 else None)

# (5) norm-matched controls -----------------------------------------------------
R["controls"] = []
for lab, cell, ctl, suffix in [
        ("Tiny ImageNet clean", "TIA", "scalar-shrink", "sclshr"),
        ("Tiny ImageNet clean", "TIA", "shuffled", "shufauto"),
        ("Tiny ImageNet 20%", "TIP", "shuffled", "shufauto"),
        ("CIFAR-10 20%", "VP", "scalar-shrink", "sclshr"),
        ("FineWeb LM", "FA", "scalar-shrink", "sclshr"),
        ("FineWeb LM", "FA", "shuffled", "shufauto")]:
    metric = "final_val" if cell == "FA" else "final_acc"
    scale = 1.0 if cell == "FA" else 100.0
    full = paired2(f"final_{cell}_auto2", f"final_{cell}_muoncos", metric, scale)
    part = paired2(f"final_{cell}_{suffix}", f"final_{cell}_muoncos", metric, scale)
    resid = paired2(f"final_{cell}_auto2", f"final_{cell}_{suffix}", metric, scale)
    if full and part:
        R["controls"].append(dict(
            cell=lab, control=ctl, full=full, control_vs_muon=part,
            echo_vs_control=resid,
            reproduced=part["delta"] / full["delta"] if full["delta"] else None))

# (6) the difficulty law ---------------------------------------------------------
LAW = [("CIFAR-10 24k", "final_B10_echomuonf", "final_B10_muoncos"),
       ("CIFAR-10 clean", "final_VAv_auto2", "final_VAv_muoncos"),
       ("CIFAR-10 20%", "final_VPv_auto2", "final_VPv_muoncos"),
       ("CIFAR-10 40%", "final_V10d40_auto2", "final_V10d40_muoncos"),
       ("CIFAR-100 24k", "final_B100_echomuonf", "final_B100_muoncos"),
       ("CIFAR-100 clean", "final_V100Am_auto2", "final_V100Av_muoncos"),
       ("CIFAR-100 20%", "final_V100Pv_auto2", "final_V100Pv_muoncos"),
       ("CIFAR-100 40%", "final_V100d40_auto2", "final_V100d40_muoncos"),
       ("Tiny ImageNet clean", "final_TIA_auto2", "final_TIA_muoncos"),
       ("Tiny ImageNet 20%", "final_TIP_auto2", "final_TIP_muoncos")]
pts, rows = [], []
for lab, ea, ma in LAW:
    r = paired2(ea, ma)
    if r:
        err = 100.0 - r["mean_b"]
        pts.append((err, r["delta"], r["se"]))
        rows.append(dict(cell=lab, baseline_error=err, margin=r["delta"], se=r["se"]))
R["difficulty_law"] = dict(fit=wls(pts) if len(pts) > 2 else None, cells=rows)

# (7) LM token-budget ladder ------------------------------------------------------
R["lm_ladder"] = dict(
    one_x=paired2("final_FA_auto2", "final_FA_muoncos", "final_val", 1.0),
    three_x_edge=paired2("final_FA3x_auto2", "final_FA3x_muoncos", "final_val", 1.0),
    three_x_corrected=paired2("final_FA3x_auto2", "final_FA3xw_muoncos", "final_val", 1.0))

# (8) byte: clean vs corrupted -----------------------------------------------------
R["byte_contrast"] = dict(
    clean_38M=paired2("final_PcleanM_auto2", "final_PcleanM_muoncos", "final_val", 1.0),
    corrupted_38M=paired2("final_PscaleM_auto2", "final_PscaleM_muoncos", "final_val", 1.0),
    corrupted_114M=paired2("final_PscaleL_auto2", "final_PscaleL_muoncos", "final_val", 1.0))

# (9) retention-horizon insensitivity ------------------------------------------------
R["horizon"] = []
for cell in ["VA", "VP", "TIA", "TIP", "FA"]:
    metric = "final_val" if cell == "FA" else "final_acc"
    scale = 1.0 if cell == "FA" else 100.0
    r = paired2(f"final_{cell}_auto2h", f"final_{cell}_auto2", metric, scale)
    if r:
        R["horizon"].append(dict(cell=cell, **r))

# (10) budget-matched control ---------------------------------------------------------
R["budget_match"] = dict(
    c10=dict(light_6k=paired2("final_VAv_auto2", "final_VAv_muoncos"),
             light_24k=paired2("final_B10_echomuonf", "final_B10_muoncos"),
             strong_24k=paired2("final_S10_echomuonf", "final_S10_muoncos")),
    c100=dict(light_6k=paired2("final_V100Am_auto2", "final_V100Av_muoncos"),
              light_24k=paired2("final_B100_echomuonf", "final_B100_muoncos"),
              strong_24k=paired2("final_S100_echomuonf", "final_S100_muoncos")))

# (11) selection noise -----------------------------------------------------------------
R["selection_noise"] = dict(
    v100a_auto2=dict(
        contiguous=pick("sweepv_V100A_auto2_lr"),
        shuffled=pick("sweepw_V100A_auto2_lr"),
        test_at_0005=(lambda s: sum(v["final_acc"] for v in s.values()) / len(s) * 100
                      if s else None)(seeds_of("final_V100Av_auto2")),
        test_at_001=(lambda s: sum(v["final_acc"] for v in s.values()) / len(s) * 100
                     if s else None)(seeds_of("final_V100Am_auto2"))),
    gaps=[dict(cell=g["cell"], arm=g["arm"],
               val_gap=g["val_selected"]["gap"] if g["val_selected"] else None,
               recheck_gap=g["split_recheck"]["gap"] if g["split_recheck"] else None)
          for g in R["grid_audit"]])

with open(os.path.join(OUT, "v3_numbers.json"), "w") as f:
    json.dump(R, f, indent=1, sort_keys=True)


# ---------------------------------------------------------------- report
def fmt(r, unit="pp"):
    if not r:
        return "(pending)"
    return ("n=%d  %+.3f %s  CI[%+.3f,%+.3f]  t=%+.2f  p=%.4f"
            % (r["n"], r["delta"], unit, r["ci"][0], r["ci"][1], r["t"], r["p"]))


print("=" * 78)
print("V3 NUMBERS  (regenerated from results/runs)")
print("=" * 78)

print("\n[1] THE 2x LEARNING-RATE RULE")
for r in R["lr_rule"]:
    print("   %-18s Muon %-8g  EchoMuon %-8g  ratio %.1fx"
          % (r["cell"], r["muon_lr"], r["echo_lr"], r["ratio"]))

print("\n[2] GRID-EDGE AUDIT (shipped vision sweeps)")
for g in R["grid_audit"]:
    s, v = g["shipped"], g["val_selected"]
    print("   %-6s %-8s shipped lr=%-7g %-9s ->  val-selected lr=%-7g (gap %+.4f)"
          % (g["cell"], g["arm"], s["lr"], s["edge"],
             v["lr"] if v else float("nan"), v["gap"] if v else float("nan")))

print("\n[3] VISION LEDGER")
for r in R["vision_ledger"]:
    print("   %-26s" % r["cell"])
    print("        shipped    %s" % fmt(r["shipped"]))
    print("        corrected  %s%s" % (fmt(r["corrected"]),
          "   [%.0f%% retained]" % (100 * r["retained"]) if r["retained"] else ""))

print("\n[4] POOLED FAMILIES (corrected lr)")
for k, v in R["pooled"].items():
    if v:
        print("   %-14s k=%d  %+.3f pp  CI[%+.3f,%+.3f]  t=%+.2f  Q=%.1f (df=%d)"
              % (k, v["k"], v["delta"], v["ci"][0], v["ci"][1], v["t"], v["Q"], v["df"]))

print("\n[5] NORM-MATCHED CONTROLS")
for c in R["controls"]:
    print("   %-22s %-14s control reproduces %5.0f%% of the gain; "
          "EchoMuon vs control t=%+.2f"
          % (c["cell"], c["control"], 100 * c["reproduced"],
             c["echo_vs_control"]["t"] if c["echo_vs_control"] else float("nan")))

print("\n[6] DIFFICULTY LAW")
f = R["difficulty_law"]["fit"]
if f:
    print("   margin(pp) = %+.4f x error(%%) %+.3f" % (f["slope"], f["intercept"]))
    print("   slope CI[%+.4f,%+.4f]  z=%+.2f   Q=%.1f (df=%d)   zero-crossing at %.1f%% error"
          % (f["ci"][0], f["ci"][1], f["z"], f["Q"], f["df"], f["x_at_zero"]))
    for c in R["difficulty_law"]["cells"]:
        print("      %-22s error %5.1f%%   margin %+.2f (se %.2f)"
              % (c["cell"], c["baseline_error"], c["margin"], c["se"]))

print("\n[7] LM TOKEN-BUDGET LADDER (nats)")
for k in ["one_x", "three_x_edge", "three_x_corrected"]:
    print("   %-20s %s" % (k, fmt(R["lm_ladder"][k], "nats")))

print("\n[8] BYTE: CLEAN vs CORRUPTED (nats)")
for k in ["clean_38M", "corrupted_38M", "corrupted_114M"]:
    print("   %-16s %s" % (k, fmt(R["byte_contrast"][k], "nats")))

print("\n[9] RETENTION-HORIZON INSENSITIVITY (true horizon - shipped)")
for h in R["horizon"]:
    print("   %-5s n=%d  %+.4f  t=%+.2f  p=%.3f" % (h["cell"], h["n"], h["delta"],
                                                    h["t"], h["p"]))

print("\n[10] BUDGET-MATCHED CONTROL")
for ds, d in R["budget_match"].items():
    print("   %s" % ds)
    for k in ["light_6k", "light_24k", "strong_24k"]:
        print("      %-12s %s" % (k, fmt(d[k])))

print("\n[11] SELECTION NOISE")
v = R["selection_noise"]["v100a_auto2"]
print("   V100A/auto2: contiguous split picks %g (gap %+.4f); shuffled picks %g (gap %+.4f)"
      % (v["contiguous"]["lr"], v["contiguous"]["gap"],
         v["shuffled"]["lr"], v["shuffled"]["gap"]))
print("   test accuracy at 0.005 = %.2f%%   at 0.01 = %.2f%%   -> consequence %.2f pp"
      % (v["test_at_0005"], v["test_at_001"], v["test_at_0005"] - v["test_at_001"]))

print("\nwrote %s" % os.path.join(OUT, "v3_numbers.json"))
