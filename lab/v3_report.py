"""Read out the v3 screening campaign (stage v3-screen) against its predictions.

Every comparison is restricted to the seeds the new arms actually ran, 1 to 3. The
published baselines have n=8 on Tiny ImageNet and n=16 on corrupted bytes, and scoring a
3-seed arm against an 8- or 16-seed baseline compares different seed sets: on seeds 1-3
the shipped method's corrupted-byte margin is +0.0004 (t=0.28), not the published +0.0010
(t=2.2). Reporting the two side by side would have manufactured an improvement out of
seed noise.
"""
import glob
import json
import math
import os

RUNS = os.path.join("results", "runs")
SEEDS = {"1", "2", "3"}


def _betacf(a, b, x):
    tiny, c, d = 1e-30, 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = tiny if abs(d) < tiny else d
    d, h = 1.0 / d, 1.0 / d
    for m in range(1, 200):
        m2 = 2 * m
        for num in (m * (b - m) * x / ((a + m2 - 1.0) * (a + m2)),
                    -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))):
            d = 1.0 + num * d
            c = 1.0 + num / c
            d = tiny if abs(d) < tiny else d
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < 3e-12:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def t_p(t, df):
    return betainc(df / 2.0, 0.5, df / (df + t * t)) if df > 0 else float("nan")


def seeds_of(prefix):
    out = {}
    for d in sorted(glob.glob(os.path.join(RUNS, prefix + "_s*"))):
        p = os.path.join(d, "final.json")
        if os.path.exists(p):
            out[os.path.basename(d).rsplit("_s", 1)[1]] = json.load(open(p))
    return out


def paired(pa, pb, metric, scale, restrict=SEEDS):
    a, b = seeds_of(pa), seeds_of(pb)
    ks = sorted((set(a) & set(b)) & restrict) if restrict else sorted(set(a) & set(b))
    ds = [(a[k][metric] - b[k][metric]) * scale for k in ks
          if a[k].get(metric) is not None and b[k].get(metric) is not None]
    if len(ds) < 2:
        return None
    n = len(ds)
    mu = sum(ds) / n
    sd = math.sqrt(sum((d - mu) ** 2 for d in ds) / (n - 1))
    se = sd / math.sqrt(n) if sd else 1e-12
    return dict(delta=mu, t=mu / se, n=n, p=t_p(mu / se, n - 1), sd=sd)


def lam(prefix):
    ls, sg = [], []
    for d in sorted(glob.glob(os.path.join(RUNS, prefix + "_s*"))):
        p = os.path.join(d, "log.jsonl")
        if not os.path.exists(p):
            continue
        if os.path.basename(d).rsplit("_s", 1)[1] not in SEEDS:
            continue
        with open(p) as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                if "gate_lambda" in j:
                    ls.append(j["gate_lambda"])
                if "signal_frac" in j:
                    sg.append(j["signal_frac"])
    return (sum(ls) / len(ls) if ls else None, min(ls) if ls else None,
            max(ls) if ls else None, sum(sg) / len(sg) if sg else None)


ARMS = [("auto2", "EchoMuon (shipped)"), ("a1abs", "abscal + gap ctl"),
        ("a2abshi", "abscal_hi + gap ctl"), ("a3sig", "normal + SIGNAL ctl"),
        ("a4both", "abscal + signal ctl")]
CELLS = [("TIA", "Tiny ImageNet clean", "final_acc", 100.0, "pp", "higher"),
         ("PscaleM", "corrupted bytes 38M", "final_val", 1.0, "nats", "lower")]

tests = []
for cell, label, metric, scale, unit, better in CELLS:
    print("=" * 78)
    print("%s   (%s, %s is better)" % (label, unit, better))
    print("=" * 78)
    print("%-22s %11s %8s %8s   %11s %8s" % ("arm", "vs Muon", "t", "p",
                                             "vs shipped", "t"))
    print("-" * 78)
    for arm, name in ARMS:
        pre = f"final_{cell}_auto2" if arm == "auto2" else f"final3_{cell}_{arm}"
        vm = paired(pre, f"final_{cell}_muoncos", metric, scale)
        ve = paired(pre, f"final_{cell}_auto2", metric, scale) if arm != "auto2" else None
        if not vm:
            print("%-22s (missing)" % name)
            continue
        if arm != "auto2":
            tests.append((f"{cell}/{arm}", vm["p"]))
        print("%-22s %+11.4f %8.2f %8.4f   %11s %8s" % (
            name, vm["delta"], vm["t"], vm["p"],
            "%+.4f" % ve["delta"] if ve else "-",
            "%.2f" % ve["t"] if ve else "-"))
    print("\n  controller")
    for arm, name in ARMS:
        pre = f"final_{cell}_auto2" if arm == "auto2" else f"final3_{cell}_{arm}"
        mu, lo, hi, sg = lam(pre)
        if mu is None:
            continue
        print("    %-22s lambda %.3f  range %.3f-%.3f%s" % (
            name, mu, lo, hi, "   signal_frac %.4f" % sg if sg is not None else ""))
    print()

tests.sort(key=lambda x: x[1])
print("Holm over the %d screening tests against Muon" % len(tests))
mx = 0.0
for i, (k, p) in enumerate(tests):
    mx = max(mx, min(1.0, p * (len(tests) - i)))
    print("   %-18s p=%.4f  p_holm=%.4f  %s"
          % (k, p, mx, "survives" if mx < 0.05 else ""))

# what n would have been needed on the byte cell
full = paired("final_PscaleM_auto2", "final_PscaleM_muoncos", "final_val", 1.0,
              restrict=None)
if full:
    need = (2.0 * full["sd"] / abs(full["delta"])) ** 2
    print("\nPower note: the published corrupted-byte effect is %+.4f nats with sd %.4f "
          "over n=%d.\nReaching |t|=2 on it needs about n=%d. This screen ran n=3, so an "
          "absence of\neffect on that cell is not evidence of a tie."
          % (full["delta"], full["sd"], full["n"], math.ceil(need)))
