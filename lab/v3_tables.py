"""Emit the paper's Appendix-B LaTeX table rows straight from results/runs.

Run:  python v3_tables.py    (writes paper_v2/v3_tables.tex and prints it)
"""
import json
import math
import os

RUNS = os.path.join("results", "runs")
OUT = "paper_v2"
T975 = {2: 4.303, 5: 2.571, 7: 2.365, 15: 2.131}


def _betacf(a, b, x):
    tiny, eps = 1e-30, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (tiny if abs(d) < tiny else d)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (tiny if abs(d) < tiny else d)
        c = tiny if abs(c) < tiny else c
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (tiny if abs(d) < tiny else d)
        c = tiny if abs(c) < tiny else c
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


def tp(t, df):
    return betainc(df / 2.0, 0.5, df / (df + t * t)) if df > 0 else float("nan")


def seeds(prefix):
    out = {}
    for d in sorted(os.listdir(RUNS)):
        if d.startswith(prefix + "_s") and d[len(prefix) + 2:].isdigit():
            p = os.path.join(RUNS, d, "final.json")
            if os.path.exists(p):
                out[int(d[len(prefix) + 2:])] = json.load(open(p))
    return out


def stat(pa, pb, metric, scale):
    A, B = seeds(pa), seeds(pb)
    common = sorted(set(A) & set(B))
    va = [A[s][metric] * scale for s in common]
    vb = [B[s][metric] * scale for s in common]
    d = [x - y for x, y in zip(va, vb)]
    n = len(d)
    md = sum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1))
    sa = math.sqrt(sum((x - sum(va) / n) ** 2 for x in va) / (n - 1))
    sb = math.sqrt(sum((x - sum(vb) / n) ** 2 for x in vb) / (n - 1))
    se = sd / math.sqrt(n)
    t = md / se
    tc = T975.get(n - 1, 2.2)
    return dict(n=n, ma=sum(va) / n, sa=sa, mb=sum(vb) / n, sb=sb, delta=md,
                lo=md - tc * se, hi=md + tc * se, t=t, p=tp(t, n - 1))


def holm(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    m = len(rows)
    run = 0.0
    for k, i in enumerate(order):
        adj = min(1.0, (m - k) * rows[i]["p"])
        run = max(run, adj)
        rows[i]["ph"] = run
    return rows


VIS = [("CIFAR-10 clean",       "final_VAv_auto2",    "final_VAv_muoncos"),
       ("CIFAR-10 noise",       "final_VPv_auto2",    "final_VPv_muoncos"),
       ("CIFAR-100 clean",      "final_V100Am_auto2", "final_V100Av_muoncos"),
       ("CIFAR-100 noise",      "final_V100Pv_auto2", "final_V100Pv_muoncos"),
       ("Tiny ImageNet clean",  "final_TIA_auto2",    "final_TIA_muoncos"),
       ("Tiny ImageNet noise",  "final_TIP_auto2",    "final_TIP_muoncos")]

LM = [("enwik8 clean 38M",     "final_PcleanM_auto2", "final_PcleanM_muoncos"),
      ("enwik8 corrupt 11M",   "final_P_auto2",       "final_P_muoncos"),
      ("enwik8 corrupt 38M",   "final_PscaleM_auto2", "final_PscaleM_muoncos"),
      ("enwik8 corrupt 114M",  "final_PscaleL_auto2", "final_PscaleL_muoncos"),
      ("FineWeb 162M $1\\times$", "final_FA_auto2",   "final_FA_muoncos"),
      ("FineWeb 162M $3\\times$", "final_FA3x_auto2", "final_FA3xw_muoncos"),
      ("Mamba-2 84.6M",        "final_FM_auto2",      "final_FM_muoncos")]


def emit(rows, fam, metric, scale, fmt):
    out = []
    got = []
    for lab, a, b in fam:
        try:
            r = stat(a, b, metric, scale)
        except (ZeroDivisionError, ValueError, KeyError):
            print("%% skipped %s (missing runs)" % lab)
            continue
        r["lab"] = lab
        got.append(r)
    holm(got)
    for r in got:
        out.append(("%s & $%s \\pm %s$ & $%s \\pm %s$ & $%+.*f$ & $[%+.*f, %+.*f]$ "
                    "& $%+.2f$ & %s & %s \\\\")
                   % (r["lab"], fmt % r["ma"], fmt % r["sa"], fmt % r["mb"], fmt % r["sb"],
                      rows, r["delta"], rows, r["lo"], rows, r["hi"], r["t"],
                      ("%.4f" % r["p"]).lstrip("0"), ("%.4f" % r["ph"]).lstrip("0")))
    return out


os.makedirs(OUT, exist_ok=True)
lines = ["%% VISION (accuracy pp, corrected selection)"]
lines += emit(2, VIS, "final_acc", 100.0, "%.2f")
lines.append("")
lines.append("%% LANGUAGE MODELS (validation loss, nats)")
lines += emit(4, LM, "final_val", 1.0, "%.4f")
txt = "\n".join(lines)
open(os.path.join(OUT, "v3_tables.tex"), "w").write(txt + "\n")
print(txt)
