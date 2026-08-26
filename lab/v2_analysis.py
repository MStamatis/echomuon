"""v2_analysis.py -- zero-compute ground truth for the EchoMuon v2 revision.

Reads results/runs/*/final.json (+ log.jsonl where needed) and prints:
  1. per-cell paired stats (n, means, SD, paired delta, 95% CI, t, exact p, Holm)
  2. mean lambda per cell (from final.json mean_lambda)
  3. step- and sample-accounting (crossings vs Muon's mean final quality)
  4. gate-mean distribution from the diag runs
  5. chosen lr + grid per cell/arm, with edge flags
  6. PscaleL/PscaleM sweep curves (edge-vs-interior check)
Writes paper_v2/v2_numbers.json with everything.
"""
import json, math, os
from collections import defaultdict

RUNS = os.path.join("results", "runs")
OUT = os.path.join("paper_v2")
os.makedirs(OUT, exist_ok=True)


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


# ---------- exact two-sided p for Student t via regularized incomplete beta ----------
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t, df):
    x = df / (df + t * t)
    return betainc(df / 2.0, 0.5, x)


T975 = {3: 3.1824, 5: 2.5706, 7: 2.3646, 15: 2.1315}


def paired(cell, arm_a, arm_b, metric, scale=1.0):
    """paired stats of arm_a - arm_b on shared seeds; returns dict or None"""
    A, B = seeds_of(f"final_{cell}_{arm_a}"), seeds_of(f"final_{cell}_{arm_b}")
    common = sorted(set(A) & set(B))
    if not common:
        return None
    va = [A[s][metric] * scale for s in common]
    vb = [B[s][metric] * scale for s in common]
    d = [x - y for x, y in zip(va, vb)]
    n = len(d)
    md = sum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    ma, mb = sum(va) / n, sum(vb) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in va) / (n - 1)) if n > 1 else float("nan")
    sb = math.sqrt(sum((x - mb) ** 2 for x in vb) / (n - 1)) if n > 1 else float("nan")
    t = md / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    df = n - 1
    p = t_two_sided_p(t, df)
    tc = T975.get(df)
    ci = (md - tc * sd / math.sqrt(n), md + tc * sd / math.sqrt(n)) if tc else (None, None)
    return dict(cell=cell, a=arm_a, b=arm_b, n=n, mean_a=ma, sd_a=sa, mean_b=mb, sd_b=sb,
                delta=md, sd_d=sd, t=t, df=df, p=p, ci=ci, per_seed=dict(zip(common, d)))


def holm(results):
    """adds holm-adjusted p in place"""
    idx = sorted(range(len(results)), key=lambda i: results[i]["p"])
    m = len(results)
    prev = 0.0
    for rank, i in enumerate(idx):
        adj = min(1.0, (m - rank) * results[i]["p"])
        adj = max(adj, prev)
        results[i]["p_holm"] = adj
        prev = adj


report = {}

# ---------- 1. paired stats ----------
VIS = ["VA", "VP", "V100A", "V100P", "TIA", "TIP"]
LM = ["P", "PscaleM", "PscaleL", "FA", "FM"]

vis_primary = [paired(c, "auto2", "muoncos", "final_acc", 100.0) for c in VIS]
vis_primary = [r for r in vis_primary if r]
holm(vis_primary)
lm_primary = [paired(c, "auto2", "muoncos", "final_val") for c in LM]
lm_primary = [r for r in lm_primary if r]
holm(lm_primary)
strong = [paired(c, "echomuonf", "muoncos", "final_acc", 100.0) for c in ["S10x", "S100x"]]
strong = [r for r in strong if r]
vs_adamw = ([paired(c, "auto2", "adamwcos", "final_acc", 100.0) for c in VIS]
            + [paired(c, "echomuonf", "adamwcos", "final_acc", 100.0) for c in ["S10x", "S100x"]]
            + [paired(c, "auto2", "adamwcos", "final_val") for c in LM])
vs_adamw = [r for r in vs_adamw if r]
fast = ([paired(c, "auto2f", "auto2", "final_acc", 100.0) for c in ["TIA", "TIP"]]
        + [paired("FA", "auto2f", "auto2", "final_val")])
fast = [r for r in fast if r]

report["vision_primary"] = vis_primary
report["lm_primary"] = lm_primary
report["strong"] = strong
report["vs_adamw"] = vs_adamw
report["fast"] = fast


def show(title, rows, unit):
    print(f"\n== {title} ==")
    for r in rows:
        ph = f" holm={r.get('p_holm'):.4f}" if "p_holm" in r else ""
        print(f"{r['cell']:8s} {r['a']}-{r['b']}: n={r['n']} "
              f"A={r['mean_a']:.4f}(sd {r['sd_a']:.4f}) B={r['mean_b']:.4f}(sd {r['sd_b']:.4f}) "
              f"D={r['delta']:+.4f}{unit} sd_D={r['sd_d']:.4f} "
              f"CI95=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] t={r['t']:+.2f} p={r['p']:.5f}{ph}")


show("VISION primary (acc pp, auto2 - muoncos)", vis_primary, "pp")
show("LM primary (nats, auto2 - muoncos)", lm_primary, "")
show("STRONG (acc pp, echomuonf - muoncos)", strong, "pp")
show("FAST profile (auto2f - auto2)", fast, "")
show("vs AdamW", vs_adamw, "")

# ---------- 2. lambda per cell ----------
lam = {}
for cell, arm in ([(c, "auto2") for c in VIS + LM] + [(c, "auto2f") for c in ["TIA", "TIP", "FA"]]
                  + [("S10x", "echomuonf"), ("S100x", "echomuonf")]):
    S = seeds_of(f"final_{cell}_{arm}")
    vals = [j.get("mean_lambda") for j in S.values() if j.get("mean_lambda") is not None]
    if vals:
        lam[f"{cell}_{arm}"] = dict(n=len(vals), mean=sum(vals) / len(vals),
                                    min=min(vals), max=max(vals))
report["lambda"] = lam
print("\n== mean lambda per cell ==")
for k, v in sorted(lam.items()):
    print(f"{k:16s} n={v['n']} mean={v['mean']:.4f} min={v['min']:.3f} max={v['max']:.3f}")

# ---------- 3. crossings + sample accounting ----------
def evals_of(run):
    rows = []
    p = os.path.join(RUNS, run, "log.jsonl")
    if not os.path.exists(p):
        return rows
    with open(p) as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            if "val_loss" in j and "step" in j:
                rows.append(j)
    return rows


def crossing(cell, echo_arm, muon_arm, mode):
    """mode: 'acc' (higher better) or 'loss'"""
    M = seeds_of(f"final_{cell}_{muon_arm}")
    key = "final_acc" if mode == "acc" else "final_val"
    target = sum(j[key] for j in M.values()) / len(M)
    out = {}
    for arm in (echo_arm, muon_arm):
        crosses = []
        for s in sorted(seeds_of(f"final_{cell}_{arm}")):
            rows = evals_of(f"final_{cell}_{arm}_s{s}")
            hit = None
            for r in rows:
                v = r.get("acc") if mode == "acc" else r.get("val_loss")
                if v is None:
                    continue
                ok = (v >= target) if mode == "acc" else (v <= target)
                if ok:
                    hit = r["step"]
                    break
            crosses.append((s, hit))
        reached = [c for _, c in crosses if c is not None]
        out[arm] = dict(target=target, per_seed=crosses,
                        n_reached=len(reached), n=len(crosses),
                        mean_cross=(sum(reached) / len(reached)) if reached else None)
    return out


cross = {}
for cell, mode in [("FA", "loss"), ("TIA", "acc"), ("TIP", "acc")]:
    cross[cell] = crossing(cell, "auto2", "muoncos", mode)
report["crossing"] = cross
print("\n== crossings (target = muoncos mean final) ==")
for cell, d in cross.items():
    for arm, v in d.items():
        print(f"{cell} {arm}: mean_cross={v['mean_cross']} reached {v['n_reached']}/{v['n']} "
              f"target={v['target']:.4f} per_seed={v['per_seed']}")

# sample accounting: probes add 8 forward batches every probe_every=100 (standard)
SA = {}
for cell, batch, block in [("FA", 16, 1024), ("TIA", 128, 1), ("TIP", 128, 1)]:
    d = cross[cell]
    e, m = d["auto2"], d["muoncos"]
    if e["mean_cross"] is None or m["mean_cross"] is None:
        continue
    unit = batch * block
    echo_ex = e["mean_cross"] * unit + (e["mean_cross"] / 100.0) * 8 * unit
    muon_ex = m["mean_cross"] * unit
    SA[cell] = dict(echo_examples=echo_ex, muon_examples=muon_ex, ratio=echo_ex / muon_ex,
                    step_ratio=e["mean_cross"] / m["mean_cross"])
    print(f"{cell}: echo_ex={echo_ex:,.0f} muon_ex={muon_ex:,.0f} "
          f"sample_ratio={echo_ex/muon_ex:.3f} step_ratio={e['mean_cross']/m['mean_cross']:.3f}")
report["sample_accounting"] = SA

# ---------- 4. gate-mean distribution from diag runs ----------
gm = []
for d in ["diag_M_tcg", "diag_S_tcg"]:
    p = os.path.join(RUNS, d, "log.jsonl")
    if not os.path.exists(p):
        continue
    with open(p) as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            for k, v in j.items():
                if "gate" in k and isinstance(v, (int, float)):
                    gm.append(float(v))
if gm:
    gm.sort()
    report["gate_mean"] = dict(n=len(gm), mean=sum(gm) / len(gm), min=gm[0], max=gm[-1],
                               ge1=sum(1 for x in gm if x >= 1.0),
                               p50=gm[len(gm) // 2])
    print(f"\n== gate means (diag runs) == n={len(gm)} mean={sum(gm)/len(gm):.4f} "
          f"min={gm[0]:.4f} max={gm[-1]:.4f} median={gm[len(gm)//2]:.4f} "
          f">=1.0: {sum(1 for x in gm if x>=1.0)}")

# ---------- 5. chosen lr + grids ----------
print("\n== chosen lr and sweep grids ==")
grids = defaultdict(list)
for d in os.listdir(RUNS):
    if "_lr" in d and d.startswith(("sweep_", "ft_sweep_")):
        base, _, lr = d.rpartition("_lr")
        j = load(d)
        if j:
            grids[base].append((float(lr), j["final_val"]))
lrsel = {}
for cell in VIS + LM + ["S10x", "S100x"]:
    for arm in ["auto2", "auto2f", "muoncos", "adamwcos", "echomuonf"]:
        S = seeds_of(f"final_{cell}_{arm}")
        if not S:
            continue
        chosen = {j["lr"] for j in S.values()}
        key = f"sweep_{cell}_{arm}"
        # strong cells swept under S10/S100 names
        g = grids.get(key) or grids.get(key.replace("S10x", "S10").replace("S100x", "S100"))
        gl = sorted(x[0] for x in g) if g else None
        edge = None
        if gl and len(chosen) == 1:
            c = next(iter(chosen))
            edge = ("BOTTOM" if c == gl[0] else "TOP" if c == gl[-1] else "interior") if c in gl else "off-grid"
        lrsel[f"{cell}_{arm}"] = dict(chosen=sorted(chosen), grid=gl, edge=edge)
        print(f"{cell:8s} {arm:10s} chosen={sorted(chosen)} grid={gl} edge={edge}")
report["lr_selection"] = lrsel

print("\n== PscaleM / PscaleL / FA sweep curves ==")
for key in ["sweep_PscaleM_auto2", "sweep_PscaleM_muoncos", "sweep_PscaleL_auto2",
            "sweep_PscaleL_muoncos", "sweep_FA_auto2", "sweep_FA_muoncos"]:
    if key in grids:
        curve = sorted(grids[key])
        print(key, " ".join(f"lr{lr:g}:{v:.4f}" for lr, v in curve))
        report.setdefault("sweep_curves", {})[key] = curve

with open(os.path.join(OUT, "v2_numbers.json"), "w") as f:
    json.dump(report, f, indent=1, default=str)
print("\nwritten paper_v2/v2_numbers.json")
