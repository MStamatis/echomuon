"""Read out the v3 campaign (stages v3-screen and v3-confirm) against its predictions.

Every row is scored on ONE seed set: the seeds that row's arm actually ran. The shipped
baseline is printed twice, at its own full n and re-scored on the arm's seeds, because
those are not the same number and treating them as one manufactures effects. On seeds
1-3 the shipped method's corrupted-byte margin is +0.0004 (t=0.28); at its full n=16 it
is +0.0010 (t=2.2). A 3-seed arm compared against the second of those looks like an
improvement it has not earned.
"""
import glob
import json
import math
import os

RUNS = os.path.join("results", "runs")


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


def paired(pa, pb, metric, scale, restrict=None):
    a, b = seeds_of(pa), seeds_of(pb)
    ks = set(a) & set(b)
    if restrict is not None:
        ks &= restrict
    ds = [(a[k][metric] - b[k][metric]) * scale for k in sorted(ks)
          if a[k].get(metric) is not None and b[k].get(metric) is not None]
    if len(ds) < 2:
        return None
    n = len(ds)
    mu = sum(ds) / n
    sd = math.sqrt(sum((d - mu) ** 2 for d in ds) / (n - 1))
    se = sd / math.sqrt(n) if sd else 1e-12
    return dict(delta=mu, t=mu / se, n=n, p=t_p(mu / se, n - 1), sd=sd)


def lam(prefix, restrict=None):
    ls, sg = [], []
    for d in sorted(glob.glob(os.path.join(RUNS, prefix + "_s*"))):
        if restrict is not None and os.path.basename(d).rsplit("_s", 1)[1] not in restrict:
            continue
        p = os.path.join(d, "log.jsonl")
        if not os.path.exists(p):
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


ARMS = [("a1abs", "abscal + gap ctl"), ("a2abshi", "abscal_hi + gap ctl"),
        ("a3sig", "normal + SIGNAL ctl"), ("a4both", "abscal + signal ctl")]
CELLS = [("TIA", "Tiny ImageNet clean", "final_acc", 100.0, "pp", "higher"),
         ("PscaleM", "corrupted bytes 38M", "final_val", 1.0, "nats", "lower")]

tests = []
for cell, label, metric, scale, unit, better in CELLS:
    muon, ship = f"final_{cell}_muoncos", f"final_{cell}_auto2"
    full = paired(ship, muon, metric, scale)
    print("=" * 92)
    print("%s   (%s, %s is better)" % (label, unit, better))
    print("=" * 92)
    if full:
        print("shipped EchoMuon at its own full n=%d:  %+.4f  t=%.2f  p=%.4f"
              % (full["n"], full["delta"], full["t"], full["p"]))
    print("-" * 92)
    print("%-22s %3s %11s %7s %8s | %13s | %12s %6s"
          % ("arm", "n", "vs Muon", "t", "p", "shipped, same", "vs shipped", "t"))
    print("-" * 92)
    for arm, name in ARMS:
        pre = f"final3_{cell}_{arm}"
        ks = set(seeds_of(pre))
        if not ks:
            continue
        vm = paired(pre, muon, metric, scale, ks)
        sm = paired(ship, muon, metric, scale, ks)     # shipped on the SAME seeds
        ve = paired(pre, ship, metric, scale, ks)
        if not vm:
            continue
        tests.append((f"{cell}/{arm}", vm["p"], vm["n"]))
        print("%-22s %3d %+11.4f %7.2f %8.4f | %+13.4f | %+12.4f %6.2f"
              % (name, vm["n"], vm["delta"], vm["t"], vm["p"],
                 sm["delta"] if sm else float("nan"),
                 ve["delta"] if ve else float("nan"),
                 ve["t"] if ve else float("nan")))
    print("\n  controller, on each arm's own seeds")
    for arm, name in ARMS:
        pre = f"final3_{cell}_{arm}"
        ks = set(seeds_of(pre))
        if not ks:
            continue
        mu, lo, hi, sg = lam(pre, ks)
        if mu is None:
            continue
        print("    %-22s lambda %.3f  range %.3f-%.3f%s"
              % (name, mu, lo, hi, "   signal_frac %.4f" % sg if sg is not None else ""))
    mu, lo, hi, _ = lam(ship)
    if mu is not None:
        print("    %-22s lambda %.3f  range %.3f-%.3f   <- shipped, full n"
              % ("EchoMuon (gap ctl)", mu, lo, hi))
    if full:
        need = max(2, math.ceil((2.0 * full["sd"] / abs(full["delta"])) ** 2)) if full["delta"] else 0
        print("\n  power: the shipped effect here is %+.4f with sd %.4f; |t|=2 needs about n=%d"
              % (full["delta"], full["sd"], need))
    print()

tests.sort(key=lambda x: x[1])
print("Holm across the %d arm-cell tests against Muon" % len(tests))
mx = 0.0
for i, (k, p, n) in enumerate(tests):
    mx = max(mx, min(1.0, p * (len(tests) - i)))
    print("   %-18s n=%-3d p=%.4f  p_holm=%.4f  %s"
          % (k, n, p, mx, "survives" if mx < 0.05 else ""))
