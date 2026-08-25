"""Idea 3: spectral recycling. Log the singular spectrum of every momentum buffer
(the information Muon computes implicitly and throws away) as a per-layer, per-step
control signal; optionally close the loop with a per-layer lr controller.

Controller modes (ablation set):
  srank    - lr scale = clip(sqrt(stable_rank / median), 0.5, 2)   (the real controller)
  inverse  - clip(sqrt(median / stable_rank), 0.5, 2)              (directionality test)
  shuffled - srank scales randomly permuted across layers          (assignment test)

Signals (spectral-ctl composition; all derived from the same amortized spectral probe):
  rank  - the stable-rank lr controller above
  conf  - soft SNR gate: multiply the layer lr by mean shrinker confidence w-bar
          (Idea 1 as a free scalar; floor 0.05 so training never fully freezes)
  conf2 - trend-gated version of conf: gate on w-bar's DECLINE from its own running
          max, not its absolute level. Stationary-low w-bar (healthy learning under
          noise) leaves lr untouched; falling w-bar (signal dying, memorization
          phase) progressively cuts it. Fixes conf's regression in noisy pretraining.
  valve - predictive safety valve: if a layer's top singular value jumps above
          2x its EMA, cut that layer's lr 10x until it recovers
"""
import json

import numpy as np
import torch

from .rmt import spectrum_stats


class SpectralMonitor:
    def __init__(self, optimizer, every: int, path: str, controller: bool = False,
                 ctl_mode: str = "srank", signals=("rank",), seed: int = 0):
        self.opt = optimizer
        self.every = max(1, every)
        self.path = path
        self.controller = controller
        self.ctl_mode = ctl_mode
        self.signals = set(signals)
        self.ema_top = {}
        self.wbar_ema = {}
        self.wbar_ref = {}
        self.rng = np.random.RandomState(seed)
        self._fh = open(path, "a", buffering=1)

    def maybe_log(self, step: int, extra: dict | None = None):
        if step % self.every != 0:
            return
        rows, stats = [], {}
        for name, buf in self.opt.matrix_momentum():
            s = torch.linalg.svdvals(buf.float()).cpu().numpy()
            rec = {"step": step, "layer": name, **spectrum_stats(s, tuple(buf.shape))}
            if name in getattr(self.opt, "last_mean_w", {}):
                rec["opt_mean_w"] = float(self.opt.last_mean_w[name])
            if name in getattr(self.opt, "last_gate_mean", {}):
                rec["gate_mean"] = self.opt.last_gate_mean[name]
            rows.append(rec)
            stats[name] = rec
        if self.controller and stats:
            scales = {name: 1.0 for name in stats}
            if "rank" in self.signals:
                med = float(np.median([r["stable_rank"] for r in stats.values()]))
                for name, r in stats.items():
                    sr = r["stable_rank"]
                    ratio = med / (sr + 1e-12) if self.ctl_mode == "inverse" else sr / (med + 1e-12)
                    scales[name] *= float(np.clip(np.sqrt(ratio), 0.5, 2.0))
                if self.ctl_mode == "shuffled":
                    names = list(scales)
                    vals = self.rng.permutation([scales[n] for n in names])
                    scales = {n: float(v) for n, v in zip(names, vals)}
            if "conf" in self.signals:
                for name, r in stats.items():
                    scales[name] *= max(r["mean_w"], 0.05)
            if "conf2" in self.signals:
                for name, r in stats.items():
                    ema = 0.8 * self.wbar_ema.get(name, r["mean_w"]) + 0.2 * r["mean_w"]
                    self.wbar_ema[name] = ema
                    ref = max(self.wbar_ref.get(name, ema), ema)
                    self.wbar_ref[name] = ref
                    scales[name] *= float(np.clip(ema / (ref + 1e-12), 0.05, 1.0))
            if "valve" in self.signals:
                for name, r in stats.items():
                    top = r["top_sv"]
                    ema = self.ema_top.get(name, top)
                    if top > 2.0 * ema:
                        scales[name] *= 0.1
                    self.ema_top[name] = 0.9 * ema + 0.1 * top
            self.opt.layer_scale = scales
        for r in rows:
            r["lr_scale"] = self.opt.layer_scale.get(r["layer"], 1.0)
        if extra:
            rows.append({"step": step, "layer": "_global", **extra})
        for r in rows:
            self._fh.write(json.dumps(r) + "\n")

    def close(self):
        self._fh.close()
