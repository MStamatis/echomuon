"""EchoMuon: Muon with a per-direction temporal trust gate and an optional
memorization-gap controller.

Muon (Jordan et al., 2024) orthogonalizes the momentum of 2-D hidden weight
matrices, giving every singular direction of the update equal trust. EchoMuon
prices that trust by each direction's *echo*: its support in a second, slower
momentum buffer. Directions whose fast/slow buffers agree keep their step;
directions with no echo are damped. Scores are referenced to the layer median
and clipped to [0.1, 1.0], so the multiplier never exceeds 1: the gate is a
CONTRACTION of the step, not a reallocation at constant norm. The measured mean
is 0.957, with 0 of 5,760 logged layer-steps at or above 1.0. It therefore does
interact with the learning rate, and EchoMuon has to be swept on its own grid
instead of sharing Muon's; the README gives the measured sensitivity. A floor
keeps every direction alive. A scalar ``gate_lambda`` in [0, 1] interpolates the
gate between OFF (exactly plain Muon) and fully ON; the
MemorizationGapController sets it from a measured old-vs-fresh loss gap.

Status: experimental. The README states the scope of the evidence and lists the
claims withdrawn since the first release.

Reference: S. Mastromichalakis, "EchoMuon: Cross-Timescale Gating in Muon, and
the Learning-Rate Confound in Gated Optimizers", 2026.
"""
from __future__ import annotations

import math

import torch


__all__ = ["EchoMuon", "MemorizationGapController", "newton_schulz5"]


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalization (Keller Jordan's coefficients)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16) if G.is_cuda else G.to(torch.float32)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class EchoMuon(torch.optim.Optimizer):
    """EchoMuon optimizer for 2-D hidden weight matrices.

    Like Muon, this optimizer is meant ONLY for the hidden matrix parameters of
    a network (attention/MLP weights). Embeddings, output heads, gains, and
    biases should be handled by a separate AdamW -- exactly as with Muon.

    Arguments:
        params: 2-D parameters to optimize (an iterable of tensors or dicts).
        lr: learning rate (Muon convention; sweep it as you would for Muon).
        momentum: fast EMA-sum decay (Muon's own buffer), default 0.95.
        slow_beta: slow buffer decay, default 0.99. The gate scores each
            singular direction of the fast buffer by its normalized support in
            this slower buffer.
        nesterov: use the Nesterov form of the fast buffer (default True).
        weight_decay: decoupled weight decay (default 0).
        ns_steps: Newton-Schulz iterations (default 5).
        gate_every: refresh the gate basis every this many steps (default 100,
            the paper's "fast profile"; 25 is the standard profile -- same
            quality within noise, ~2x the gate overhead).
        gate_floor: minimum gate value so no direction is fully silenced
            (default 0.1).
        gate_lambda: gate strength in [0, 1]. 0 = exactly plain Muon; 1 = full
            gate. Set it directly, or let MemorizationGapController drive it
            from a measured memorization gap.

    Example::

        hidden = [p for n, p in model.named_parameters()
                  if p.ndim == 2 and "embed" not in n and "head" not in n]
        others = [p for n, p in model.named_parameters() if p not in set(hidden)]
        opt = EchoMuon(hidden, lr=0.02)
        aux = torch.optim.AdamW(others, lr=3e-4, weight_decay=0.1)
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 slow_beta: float = 0.99, nesterov: bool = True,
                 weight_decay: float = 0.0, ns_steps: int = 5,
                 gate_every: int = 100, gate_floor: float = 0.1,
                 gate_lambda: float = 1.0):
        if not 0.0 <= gate_lambda <= 1.0:
            raise ValueError("gate_lambda must be in [0, 1]")
        defaults = dict(lr=lr, momentum=momentum, slow_beta=slow_beta,
                        nesterov=nesterov, weight_decay=weight_decay,
                        ns_steps=ns_steps, gate_every=max(1, gate_every),
                        gate_floor=gate_floor)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        "EchoMuon only accepts 2-D parameters; route "
                        f"{tuple(p.shape)} tensors to AdamW instead.")
        self.gate_lambda = float(gate_lambda)
        self._t = 0

    @torch.no_grad()
    def _refresh_gate(self, p, group):
        """Recompute the temporal-consistency gate for one parameter.

        For each singular direction u_i of the fast buffer M1 (via the Gram
        eigendecomposition on the small side), consistency
        c_i = (u_i^T M2 M1^T u_i) / sigma_i^2 is the slow buffer's relative
        support along that direction. Scores are referenced to the layer median
        and clipped to [gate_floor, 1.0], so the multiplier is a contraction: it
        damps low-echo directions and leaves the rest at 1.
        """
        st = self.state[p]
        M1 = st["buf"].float()
        M2 = st["buf2"].float()
        if M1.size(0) > M1.size(1):
            M1, M2 = M1.T, M2.T
        G = M1 @ M1.T
        evals, U = torch.linalg.eigh(G)
        evals = evals.clamp_min(1e-12)
        C = U.T @ (M2 @ M1.T) @ U
        c = C.diagonal() / evals
        med = torch.quantile(c, 0.5).clamp_min(1e-12)
        g = (c / med).clamp(group["gate_floor"], 1.0)
        st["gate_U"], st["gate_g"] = U, g

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._t += 1
        for group in self.param_groups:
            beta = group["momentum"]
            refresh = (self._t % group["gate_every"] == 0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                    st["buf2"] = torch.zeros_like(p)
                # Refresh from the buffers as they stood at the end of the previous
                # step, BEFORE this step's gradient is folded in. That is the order
                # the paper's runs used; tests/test_parity.py pins the two together.
                if refresh:
                    self._refresh_gate(p, group)
                buf, buf2 = st["buf"], st["buf2"]
                buf.mul_(beta).add_(p.grad)
                buf2.mul_(group["slow_beta"]).add_(p.grad)
                u = p.grad.add(buf, alpha=beta) if group["nesterov"] else buf
                d = newton_schulz5(u, group["ns_steps"])
                if "gate_U" in st and self.gate_lambda > 0.0:
                    U, g = st["gate_U"], st["gate_g"]
                    g_eff = 1.0 - self.gate_lambda * (1.0 - g)
                    transposed = d.size(0) > d.size(1)
                    O = (d.T if transposed else d).float()
                    O = O - U @ ((1.0 - g_eff).unsqueeze(1) * (U.T @ O))
                    d = (O.T if transposed else O).to(d.dtype)
                d = d * math.sqrt(max(1.0, p.size(0) / p.size(1)))
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(d, alpha=-group["lr"])
        return loss

    def state_dict(self):
        sd = super().state_dict()
        sd["echomuon_t"] = self._t
        sd["echomuon_lambda"] = self.gate_lambda
        return sd

    def load_state_dict(self, state_dict):
        self._t = state_dict.pop("echomuon_t", 0)
        self.gate_lambda = state_dict.pop("echomuon_lambda", self.gate_lambda)
        super().load_state_dict(state_dict)


class MemorizationGapController:
    """Drives ``EchoMuon.gate_lambda`` from a measured short-horizon retention gap.

    The signal: re-evaluate, under the CURRENT weights, the oldest of the last
    few training batches (with the example below, batches trained on 4-7 steps
    earlier), alongside fresh batches. The gap (fresh loss - re-seen loss) is a
    short-horizon retention signal -- how much of the last handful of updates
    has not yet spread to the rest of the data distribution;
    lambda = clip(gap / (target_frac * fresh_loss), 0, 1), EMA-smoothed. When
    the gap vanishes EchoMuon relaxes to exactly plain Muon.

    The controller does not run the forward passes itself -- your training
    loop supplies the two loss values (this keeps the package framework-free).

    Example::

        ctl = MemorizationGapController(opt)          # opt is an EchoMuon
        ring = collections.deque(maxlen=8)            # your recent batches
        for step, batch in enumerate(loader):
            train_step(batch); ring.append(batch)
            if ctl.due(step):
                with torch.no_grad():
                    reseen = mean_loss(model, list(ring)[:4])   # oldest 4
                    fresh  = mean_loss(model, next_fresh_batches(4))
                ctl.update(fresh_loss=fresh, reseen_loss=reseen)

    Arguments:
        optimizer: the EchoMuon instance whose gate_lambda is controlled.
        probe_every: probe cadence in steps (default 200, the fast profile;
            100 is the standard profile).
        target_frac: gap normalizer as a fraction of the fresh loss
            (default 0.02 -- fixed a priori in the paper, never retuned).
        ema: smoothing of lambda across probes (default 0.7).
    """

    def __init__(self, optimizer: EchoMuon, probe_every: int = 200,
                 target_frac: float = 0.02, ema: float = 0.7):
        self.opt = optimizer
        self.probe_every = max(1, probe_every)
        self.target_frac = target_frac
        self.ema = ema
        self._lam = optimizer.gate_lambda

    def due(self, step: int) -> bool:
        return step > 0 and step % self.probe_every == 0

    def update(self, fresh_loss: float, reseen_loss: float) -> float:
        gap = float(fresh_loss) - float(reseen_loss)
        target = self.target_frac * max(float(fresh_loss), 1e-12)
        lam_now = min(max(gap / target, 0.0), 1.0)
        self._lam = self.ema * self._lam + (1.0 - self.ema) * lam_now
        self.opt.gate_lambda = self._lam
        return self._lam

    @property
    def gate_lambda(self) -> float:
        return self._lam
