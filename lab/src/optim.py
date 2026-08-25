"""Hybrid optimizer: hidden 2-D weights get {muon | shrunk} updates, everything else AdamW.
mode="adamw" routes all parameters to AdamW (the tuned baseline)."""
import math

import torch

from .rmt import shrink_weights_torch


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Keller Jordan's quintic Newton-Schulz orthogonalization."""
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


def shrink_dualize_batch(Ms: list, mix: float = 0.0):
    """RMT shrinkage for a batch of same-shape matrices, fully on device (no host syncs).

    Via the Gram matrix: eigh(M M^T) on the small side gives U and S at a fraction of
    full-SVD cost, and U diag(w) V^T = U diag(w/S) U^T M. Bulk directions get w=0 and
    drop out, so the squared conditioning of the Gram matrix only touches directions
    we discard anyway. Returns (list of directions, per-matrix mean_w tensor)."""
    X = torch.stack([m.float() for m in Ms])
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)
    G = X @ X.transpose(1, 2)
    evals, U = torch.linalg.eigh(G)
    S = evals.flip(-1).clamp_min(0).sqrt()
    U = U.flip(-1)
    W = shrink_weights_torch(S, tuple(X.shape[-2:]))
    if mix > 0:
        W = mix + (1 - mix) * W
    coef = W / S.clamp_min(1e-8)
    D = (U * coef.unsqueeze(1)) @ (U.transpose(1, 2) @ X)
    if transposed:
        D = D.transpose(1, 2)
    return [d.to(Ms[0].dtype) for d in D.unbind(0)], W.mean(dim=1)


class HybridOptimizer:
    """mode="tcg" + auto-gate = EchoMuon: Muon whose update trusts each singular
    direction in proportion to its echo in the gradient history, scaled by a measured
    memorization gap. Temporal-Consistency Gating: keeps Muon's update but damps singular
    directions of the fast momentum that have no support in a slow momentum buffer.
    Persistent learning signal lives in both buffers; inconsistent supervision (label
    noise, corrupted data) flickers in the fast buffer only. Gates are median-normalized
    per layer, so they REALLOCATE trust across directions (mean ~1) instead of scaling
    the overall lr — structurally orthogonal to lr scheduling. The gate basis is
    refreshed every gate_every steps (amortized eigh); the per-step path is matmul-only
    via a cached damping projector."""

    def __init__(self, model, mode: str, lr: float, aux_lr: float = 2e-3,
                 momentum: float = 0.95, nesterov: bool = True, wd: float = 0.0,
                 aux_wd: float = 0.1, ns_steps: int = 5, shrink_mix: float = 0.0,
                 slow_beta: float = 0.99, gate_every: int = 25, gate_floor: float = 0.1,
                 gate_mode: str = "normal", gate_quantile: float = 0.5, seed: int = 0):
        # gate_quantile: reference quantile of the consistency scores. 0.5 (median)
        # touches half the directions; 0.2 damps only the most-inconsistent quintile —
        # avoids taxing the slow-emerging genuine features that dominate at larger scale.
        self.gate_quantile = gate_quantile
        # gate_block: aggregate consistency over eigenvalue-ordered blocks of this many
        # directions (energy-weighted) before gating — block means stay well-estimated
        # at any dimension, unlike single-direction scores (the H1 scale fix).
        self.gate_block = 1
        # gate_stage: "post" damps the orthogonalized update (breaks exact orthogonality);
        # "pre" damps the momentum BEFORE Newton-Schulz so the final update stays a clean
        # orthogonal matrix (the H3 scale fix).
        self.gate_stage = "post"
        assert mode in ("adamw", "muon", "shrunk", "tcg")
        assert gate_mode in ("normal", "inverse", "shuffled", "novelty", "coherence",
                             "amplify", "magnitude", "cosgate", "cautious", "mix")
        self.gate_mode = gate_mode
        self._gate_rng = torch.Generator().manual_seed(seed + 777)
        # EchoMuon controller: measured-overfitting interpolation between pure Muon (0)
        # and the full gate (1); the training loop sets this from the memorization gap.
        self.gate_lambda = 1.0
        self.mode = mode
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.wd = wd
        self.ns_steps = ns_steps
        self.shrink_mix = shrink_mix

        hidden = set(model.hidden_matrix_names()) if mode != "adamw" else set()
        self.matrix_params = [(n, p) for n, p in model.named_parameters() if n in hidden]
        aux_params = [p for n, p in model.named_parameters() if n not in hidden]
        aux_rate = lr if mode == "adamw" else aux_lr
        self.aux = torch.optim.AdamW(aux_params, lr=aux_rate, betas=(0.9, 0.95), weight_decay=aux_wd)
        self.buf = {n: torch.zeros_like(p) for n, p in self.matrix_params}
        self.layer_scale = {}       # optional per-layer lr multipliers (Idea 3 controller)
        self.last_mean_w = {}       # shrink diagnostics for the monitor
        self.slow_beta = slow_beta
        self.gate_every = max(1, gate_every)
        self.gate_floor = gate_floor
        self._t = 0
        if mode == "tcg":
            self.buf2 = {n: torch.zeros_like(p) for n, p in self.matrix_params}
            self._gate = {}         # name -> (U, g) cached damping basis
            self.last_gate_mean = {}
            if gate_mode in ("coherence", "amplify"):
                # CCG: within-window sign coherence of raw-gradient projections onto a
                # frozen singular basis. coherence = |sum s| / sum |s| — the frequency
                # axis cancels in the ratio, so rare-but-genuine features (few, coherent
                # firings) pass untouched while corrupted supervision (random signs when
                # firing) is damped. Directions that never fired are exempt.
                self._proj = {}   # name -> (U, W) frozen projection maps
                self._acc = {}    # name -> [S_sum, S_abs] per-direction accumulators
            if gate_mode == "novelty":
                # third, slower buffer: a genuinely-new direction shows RISING support
                # (recent > old); noise flickers equally at all timescales. Damp only
                # directions that are both low-support and non-growing.
                self.buf3 = {n: torch.zeros_like(p) for n, p in self.matrix_params}
                self.novelty_beta = 0.999
            if gate_mode == "cosgate":
                # Magma-style ablation (arXiv:2602.15322): one scalar per matrix,
                # EMA-smoothed sigmoid(cossim(momentum, current grad)/2) — agreement
                # against the INSTANTANEOUS gradient, not between two smoothed buffers.
                self._cos_s = {n: 0.5 for n, _ in self.matrix_params}
            # AdEMAMix-style ablation (arXiv:2409.03137): orthogonalize a linear
            # MIXTURE of the fast and slow buffers instead of gating by their
            # agreement. alpha=1 mixes them at equal EMA-normalized magnitude.
            self.gate_mix_alpha = 1.0

    @torch.no_grad()
    def step(self):
        self.aux.step()
        self._t += 1
        if self.mode == "muon":
            for name, p in self.matrix_params:
                if p.grad is None:
                    continue
                d = newton_schulz5(self._nesterov_update(name, p), self.ns_steps)
                self._apply(name, p, d)
            return
        if self.mode == "tcg":
            if self._t % self.gate_every == 0:
                self._refresh_gates()
            for name, p in self.matrix_params:
                if p.grad is None:
                    continue
                self.buf2[name].mul_(self.slow_beta).add_(p.grad)
                if self.gate_mode == "novelty":
                    self.buf3[name].mul_(self.novelty_beta).add_(p.grad)
                if self.gate_mode in ("coherence", "amplify") and name in self._proj:
                    U, W = self._proj[name]
                    Gm = p.grad.float()
                    if Gm.size(0) > Gm.size(1):
                        Gm = Gm.T
                    s = ((U.T @ Gm) * W.T).sum(dim=1)
                    self._acc[name][0] += s
                    self._acc[name][1] += s.abs()
                u = self._nesterov_update(name, p)
                if self.gate_mode == "mix":
                    # equal-magnitude mixture: buf2's EMA sum is 1/(1-beta_slow) times
                    # larger in scale than buf's, so rescale before adding. NS only
                    # sees the direction, so no lr correction is needed.
                    scale = self.gate_mix_alpha * (1 - self.slow_beta) / (1 - self.momentum)
                    u = u + scale * self.buf2[name]
                gate = self._gate.get(name)
                if gate is not None and self.gate_stage == "pre":
                    U, g = gate
                    transposed = u.size(0) > u.size(1)
                    M = (u.T if transposed else u).float()
                    M = M - U @ ((1.0 - g).unsqueeze(1) * (U.T @ M))
                    u = (M.T if transposed else M).to(u.dtype)
                d = newton_schulz5(u, self.ns_steps)
                if self.gate_mode == "cosgate":
                    b, gr = self.buf[name].float(), p.grad.float()
                    cs = (b * gr).sum() / (b.norm() * gr.norm() + 1e-12)
                    s = 0.9 * self._cos_s[name] + 0.1 * float(torch.sigmoid(cs / 2.0))
                    self._cos_s[name] = s
                    self.last_gate_mean[name] = s
                    d = d * (1.0 - self.gate_lambda * (1.0 - s))
                if self.gate_mode == "cautious":
                    # Cautious-Muon (arXiv:2411.16085): zero coordinates where the
                    # orthogonalized update disagrees in sign with the current grad,
                    # rescaled so the mask has mean ~1; lambda interpolates to plain.
                    m = ((d.float() * p.grad.float()) > 0).float()
                    d_c = (d.float() * m * (m.numel() / m.sum().clamp_min(1.0))).to(d.dtype)
                    self.last_gate_mean[name] = float(m.mean())
                    d = d + self.gate_lambda * (d_c - d)
                if gate is not None and self.gate_stage == "post":
                    U, g = gate
                    if self.gate_lambda < 1.0:
                        g = 1.0 - self.gate_lambda * (1.0 - g)
                    transposed = d.size(0) > d.size(1)
                    O = (d.T if transposed else d).float()
                    O = O - U @ ((1.0 - g).unsqueeze(1) * (U.T @ O))
                    d = (O.T if transposed else O).to(d.dtype)
                self._apply(name, p, d)
            return
        # shrunk: batch same-shape matrices -> one eigh per shape group, zero host syncs
        groups = {}
        for name, p in self.matrix_params:
            if p.grad is not None:
                groups.setdefault(tuple(p.shape), []).append((name, p))
        for shape, items in groups.items():
            us = [self._nesterov_update(name, p) for name, p in items]
            ds, mean_w = shrink_dualize_batch(us, self.shrink_mix)
            for (name, p), d, mw in zip(items, ds, mean_w.unbind(0)):
                self.last_mean_w[name] = mw  # 0-dim tensor; sync deferred to monitor
                self._apply(name, p, d)

    @torch.no_grad()
    def _refresh_gates(self):
        """Recompute the temporal-consistency gate basis: for each singular direction
        u_i of the fast buffer M1, consistency c_i = (u_i^T M2 M1^T u_i) / sigma_i^2 —
        the slow buffer's relative support along that direction. Gates are median-
        normalized per layer so they redistribute rather than rescale."""
        if self.gate_mode in ("cosgate", "cautious", "mix"):
            return  # these ablation arms use no cached singular-basis gate
        for name, p in self.matrix_params:
            M1 = self.buf[name].float()
            M2 = self.buf2[name].float()
            if M1.size(0) > M1.size(1):
                M1, M2 = M1.T, M2.T
            G = M1 @ M1.T
            evals, U = torch.linalg.eigh(G)
            evals = evals.clamp_min(1e-12)
            C = U.T @ (M2 @ M1.T) @ U
            c = C.diagonal() / evals
            if self.gate_mode == "magnitude":
                # Soft-Muon/Pion-style ablation (jiakai.xyz 2026; arXiv:2605.19282):
                # identical scaffolding (median-normalize, floor, cached projector,
                # lambda), but the gate driver is singular MAGNITUDE, not agreement.
                c = evals.sqrt()
            if self.gate_block > 1:  # energy-weighted block aggregation (ascending eigs)
                B = self.gate_block
                w = evals
                for lo in range(0, c.numel(), B):
                    hi = min(lo + B, c.numel())
                    blk = (c[lo:hi] * w[lo:hi]).sum() / w[lo:hi].sum().clamp_min(1e-12)
                    c[lo:hi] = blk
            if self.gate_mode in ("coherence", "amplify"):
                # 1) turn the last window's accumulators into gates (old basis)
                if name in self._acc and name in self._proj:
                    U_old = self._proj[name][0]
                    S_sum, S_abs = self._acc[name]
                    fired = S_abs > 1e-10
                    if int(fired.sum()) >= 8:
                        coh = S_sum.abs() / (S_abs + 1e-12)
                        med = coh[fired].median().clamp_min(1e-12)
                        g = torch.ones_like(coh)
                        if self.gate_mode == "amplify":
                            # the untried polarity: boost coherent directions (Adam-like
                            # tail amplification), never suppress; renormalize to mean 1
                            # so total update energy is preserved (schedule-orthogonal)
                            g[fired] = (coh[fired] / med).clamp(1.0, 2.0)
                            g = g / g.mean().clamp_min(1e-12)
                        else:
                            g[fired] = (coh[fired] / med).clamp(self.gate_floor, 1.0)
                        self._gate[name] = (U_old, g)
                        self.last_gate_mean[name] = float(g.mean())
                # 2) freeze a fresh basis + projection map, reset accumulators
                sig = evals.sqrt().clamp_min(1e-8)
                W = (M1.T @ U) / sig.unsqueeze(0)
                self._proj[name] = (U, W)
                self._acc[name] = [torch.zeros_like(sig), torch.zeros_like(sig)]
                continue
            if self.gate_mode == "novelty":
                M3 = self.buf3[name].float()
                if self.buf[name].size(0) > self.buf[name].size(1):
                    M3 = M3.T
                C3 = U.T @ (M3 @ M1.T) @ U
                c_slow = C3.diagonal() / evals
                c_n = c / c.median().clamp_min(1e-12)
                cs_n = c_slow / c_slow.median().clamp_min(1e-12)
                low = c_n < torch.quantile(c_n, max(self.gate_quantile, 0.05))
                not_growing = c_n < 1.2 * cs_n
                g = torch.ones_like(c_n)
                damp = low & not_growing
                g[damp] = c_n[damp].clamp(self.gate_floor, 1.0)
                self._gate[name] = (U, g)
                self.last_gate_mean[name] = float(g.mean())
                continue
            med = torch.quantile(c, self.gate_quantile).clamp_min(1e-12)
            if self.gate_mode == "inverse":  # damp CONSISTENT directions (must hurt)
                g = (med / c.clamp_min(1e-12)).clamp(self.gate_floor, 1.0)
            else:
                g = (c / med).clamp(self.gate_floor, 1.0)
            if self.gate_mode == "shuffled":  # break the direction<->gate assignment
                perm = torch.randperm(g.numel(), generator=self._gate_rng)
                g = g[perm.to(g.device)]
            self._gate[name] = (U, g)
            self.last_gate_mean[name] = float(g.mean())

    def _nesterov_update(self, name, p):
        buf = self.buf[name]
        buf.mul_(self.momentum).add_(p.grad)
        return p.grad.add(buf, alpha=self.momentum) if self.nesterov else buf

    def _apply(self, name, p, d):
        d = d * math.sqrt(max(1.0, p.size(0) / p.size(1)))
        if self.wd:
            p.mul_(1 - self.lr * self.wd)
        p.add_(d, alpha=-self.lr * self.layer_scale.get(name, 1.0))

    def zero_grad(self):
        self.aux.zero_grad(set_to_none=True)
        for _, p in self.matrix_params:
            p.grad = None

    def matrix_momentum(self):
        """Yield (name, momentum-like buffer) for 2-D hidden weights, any mode."""
        if self.mode != "adamw":
            yield from self.buf.items()
            return
        for group in self.aux.param_groups:
            for p in group["params"]:
                if p.ndim == 2 and p in self.aux.state and "exp_avg" in self.aux.state[p]:
                    name = self._aux_names.get(id(p), "")
                    if name.startswith("blocks."):
                        yield name, self.aux.state[p]["exp_avg"]

    def attach_names(self, model):
        self._aux_names = {id(p): n for n, p in model.named_parameters()}

    def state_dict(self):
        sd = {"aux": self.aux.state_dict(), "buf": self.buf,
              "layer_scale": self.layer_scale, "t": self._t}
        if self.mode == "tcg":
            sd["buf2"] = self.buf2
            if self.gate_mode == "novelty":
                sd["buf3"] = self.buf3
            if self.gate_mode == "cosgate":
                sd["cos_s"] = dict(self._cos_s)
        return sd

    def load_state_dict(self, sd):
        self.aux.load_state_dict(sd["aux"])
        for name, v in sd["buf"].items():
            self.buf[name].copy_(v.to(self.buf[name].device))
        self.layer_scale = dict(sd["layer_scale"])
        self._t = sd.get("t", 0)
        if self.mode == "tcg" and "buf2" in sd:
            for name, v in sd["buf2"].items():
                self.buf2[name].copy_(v.to(self.buf2[name].device))
            if "buf3" in sd:
                for name, v in sd["buf3"].items():
                    self.buf3[name].copy_(v.to(self.buf3[name].device))
            if "cos_s" in sd:
                self._cos_s = dict(sd["cos_s"])
            self._refresh_gates()  # gates are deterministic from the buffers
