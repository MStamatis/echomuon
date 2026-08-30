"""Step 1 of the v2 gate work: measure the real consistency-score distribution.

A synthetic replay of the shipped gate path showed the score c separates planted signal
from noise almost perfectly (4.96 against 1.66 at snr 0.5) while the median-relative
normalisation flattens that to 1.000 against 0.974. That was a stationary, fixed-direction
stream. Real gradients rotate. This measures c on real runs, so we know whether an
absolutely-calibrated gate has anything to grip before spending a campaign on it.

Analytic reference points for the shipped, unnormalised EMA-sum convention
(M1 = sum b1^k g, M2 = sum b2^k g), derived and checked numerically:

    pure iid noise      c -> (1 - b1^2) / (1 - b1*b2) = 1.6387   at b1=.95, b2=.99
    perfectly consistent c -> (1 - b1) / (1 - b2)     = 5.0000

NOTE: lab/src/optim.py is deliberately NOT touched. It is pinned to the shipped package
by tests/test_parity.py and unlocking it for logging would void that. This monkey-patches
_refresh_gates at runtime, recomputes c from the same buffers the real call used, and
cross-checks its own arithmetic against the optimiser's stored last_gate_mean. If the
CHECK column below is not "ok", the diagnostic is measuring something other than the gate
and its numbers should be discarded.
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/lab")
import src.optim as O          # noqa: E402
import src.train as T          # noqa: E402

B1, B2 = 0.95, 0.99
NOISE_C = (1 - B1 ** 2) / (1 - B1 * B2)      # 1.6387
SIGNAL_C = (1 - B1) / (1 - B2)               # 5.0000
MID = 0.5 * (NOISE_C + SIGNAL_C)             # 3.3193
FLOOR = 0.1

MAX_REFRESH = int(os.environ.get("C_PROBE_REFRESHES", "24"))
OUT = os.environ.get("C_PROBE_OUT", "/lab/results/c_probe")

_records = []
_orig = O.HybridOptimizer._refresh_gates


class _Done(Exception):
    """Stop the run once enough refreshes have been observed."""


def _pct(t, qs):
    return [float(torch.quantile(t, q)) for q in qs]


def _patched(self):
    _orig(self)                                  # the real refresh, untouched
    if self.mode != "tcg" or self.gate_mode != "normal":
        return
    per_layer = []
    for name, p in self.matrix_params:
        M1 = self.buf[name].float()
        M2 = self.buf2[name].float()
        if M1.size(0) > M1.size(1):
            M1, M2 = M1.T, M2.T
        G = M1 @ M1.T
        ev, U = torch.linalg.eigh(G)
        ev = ev.clamp_min(1e-12)
        c = (U.T @ (M2 @ M1.T) @ U).diagonal() / ev

        med = torch.quantile(c, 0.5).clamp_min(1e-12)
        g_med = (c / med).clamp(FLOOR, 1.0)
        r = ((c - NOISE_C) / (SIGNAL_C - NOISE_C)).clamp(0.0, 1.0)
        g_abs = FLOOR + (1.0 - FLOOR) * r

        # eigenvalues ascend out of eigh; the top decile by singular energy is the tail
        k = max(1, c.numel() // 10)
        top, bot = c[-k:], c[:c.numel() // 2]

        # does our arithmetic reproduce what the optimiser itself stored?
        stored = self.last_gate_mean.get(name)
        ok = stored is not None and abs(float(g_med.mean()) - float(stored)) < 1e-6

        p = _pct(c, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
        per_layer.append(dict(
            n=int(c.numel()), check=bool(ok),
            c_p01=p[0], c_p10=p[1], c_p25=p[2], c_p50=p[3],
            c_p75=p[4], c_p90=p[5], c_p99=p[6], c_max=float(c.max()),
            c_top10pct=float(top.mean()), c_bottom50pct=float(bot.mean()),
            frac_above_mid=float((c > MID).float().mean()),
            frac_above_noise=float((c > NOISE_C * 1.25).float().mean()),
            med_gate_mean=float(g_med.mean()), med_gate_std=float(g_med.std()),
            abs_gate_mean=float(g_abs.mean()), abs_gate_std=float(g_abs.std()),
            abs_gate_p10=float(torch.quantile(g_abs, 0.10)),
            abs_gate_p90=float(torch.quantile(g_abs, 0.90)),
        ))
    _records.append(dict(t=self._t, layers=per_layer))
    if len(_records) >= MAX_REFRESH:
        raise _Done()


O.HybridOptimizer._refresh_gates = _patched


CELLS = {
    # name: (verdict in the paper, argv tail). Steps stay at the published value so the
    # cosine schedule is identical to the real run; we stop early by exception instead.
    "TIA_win": ("EchoMuon +1.76pp, biggest win", [
        "--task", "vision", "--dataset", "tinyimagenet", "--lr", "0.01",
        "--batch", "128", "--steps", "6000", "--n-layer", "6", "--n-head", "4",
        "--dim", "256"]),
    "VA_ns": ("EchoMuon +0.26pp, not significant", [
        "--task", "vision", "--dataset", "cifar10", "--lr", "0.005",
        "--batch", "128", "--steps", "6000", "--n-layer", "6", "--n-head", "4",
        "--dim", "256"]),
    "PscaleM_loss": ("EchoMuon LOSES, t=+2.2", [
        "--task", "lm", "--dataset", "enwik8p10", "--lr", "0.02",
        "--batch", "64", "--steps", "4000", "--n-layer", "12", "--n-head", "8",
        "--dim", "512", "--block", "256"]),
    "FA_win": ("EchoMuon -0.031 nats, LM win", [
        "--task", "lm", "--dataset", "fineweb", "--lr", "0.02",
        "--batch", "16", "--steps", "3000", "--n-layer", "12", "--n-head", "12",
        "--dim", "768", "--block", "1024"]),
}

COMMON = ["--optimizer", "tcg", "--gate-mode", "normal", "--auto-gate",
          "--auto-version", "2", "--gate-every", "25", "--probe-every", "100",
          "--lr-schedule", "cosine", "--seed", "1", "--no-monitor",
          "--data-dir", "/lab/data"]

which = sys.argv[1] if len(sys.argv) > 1 else None
os.makedirs(OUT, exist_ok=True)

for name, (verdict, tail) in CELLS.items():
    if which and which != name:
        continue
    _records.clear()
    sys.argv = ["train", "--out", "/tmp/cprobe_out", "--run-id", "cprobe_" + name] + COMMON + tail
    print("\n=== %s (%s) ===" % (name, verdict), flush=True)
    try:
        T.main()
    except _Done:
        pass
    except Exception as e:                                   # noqa: BLE001
        print("  FAILED: %s: %s" % (type(e).__name__, e), flush=True)
        continue
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(dict(cell=name, verdict=verdict, noise_c=NOISE_C, signal_c=SIGNAL_C,
                       records=_records), f)
    flat = [L for r in _records for L in r["layers"]]
    if not flat:
        print("  no refreshes captured", flush=True)
        continue

    def m(k):
        return sum(L[k] for L in flat) / len(flat)

    print("  refreshes %d over %d layer-observations   CHECK: %s"
          % (len(_records), len(flat),
             "ok" if all(L["check"] for L in flat) else "MISMATCH"), flush=True)
    print("  c:  p10 %.2f   p50 %.2f   p90 %.2f   p99 %.2f   max %.2f"
          % (m("c_p10"), m("c_p50"), m("c_p90"), m("c_p99"), m("c_max")), flush=True)
    print("      top-10%% of directions %.2f   bottom-50%% %.2f"
          % (m("c_top10pct"), m("c_bottom50pct")), flush=True)
    print("      above midpoint %.4f = %.1f%% of directions"
          % (MID, 100 * m("frac_above_mid")), flush=True)
    print("  gate: median-relative mean %.4f (sd %.4f)   <- ships"
          % (m("med_gate_mean"), m("med_gate_std")), flush=True)
    print("        absolute        mean %.4f (sd %.4f)  p10 %.3f  p90 %.3f"
          % (m("abs_gate_mean"), m("abs_gate_std"),
             m("abs_gate_p10"), m("abs_gate_p90")), flush=True)
