# EchoMuon

**Code, runs and audit for the preprint [*EchoMuon: Cross-Timescale Gating in Muon, and the Learning-Rate Confound in Gated Optimizers*](https://doi.org/10.20944/preprints202608.2150.v1)** (Preprints.org, 2026).

> ## Experimental status — read this first
>
> **This is a research prototype at an experimental stage, not a production optimizer.**
>
> The claims in earlier versions of this package were substantially overstated. A
> systematic audit of our own results (1,815 runs, all released) found that a large
> part of the originally reported advantage was an artifact of **learning-rate
> selection**, and cut the vision numbers by 42–80%. The mechanism is real and
> survives norm-matched controls on one benchmark; its useful regime is narrow, and
> both facts are documented below. **Do not adopt this in place of a properly tuned
> Muon or AdamW baseline without measuring it on your own task.** It is also
> **single-GPU only**.

[Muon](https://kellerjordan.github.io/posts/muon/) orthogonalizes the momentum of 2-D
hidden weight matrices, giving every singular direction of the update equal trust.
EchoMuon prices that trust by each direction's *echo* — its support in a second,
slower momentum buffer (β=0.99 next to Muon's 0.95). Persistent signal appears in
both buffers; noise flickers in the fast one and leaves little trace in the slow one.

The gate scores each singular direction by cross-timescale **agreement**,
`c_i = uᵢᵀ M₂M₁ᵀ uᵢ / σᵢ²` — *not* by magnitude, so a large direction sustained by a
few noisy batches is damped while a small persistent one is trusted. Scores are
referenced to the layer median and clipped to `[0.1, 1.0]`, then engaged in
proportion to a measured retention gap λ ∈ [0, 1].

### What the gate actually does to the step

Earlier documentation described the gate as mean-1, and therefore unable to act as a
disguised learning-rate schedule. The logs say otherwise: the multiplier is clipped at
1.0 and never exceeds it (0 of 5,760 logged layer-steps), with mean ≈ 0.957. It is a
**contraction**, not a reallocation.

The measurable consequence is that **EchoMuon sits higher on the learning-rate axis than
Muon**. Normalising each arm to its own optimum, across four cells with 3-seed sweeps:

| learning rate | Muon penalty | EchoMuon penalty | |
|---|---|---|---|
| **half** the optimum | +0.072 | **+0.113** | EchoMuon 1.6× worse |
| **double** the optimum | +0.063 | **+0.017** | EchoMuon 3.7× better |

This holds in all four cells and in both directions. The window has **shifted up, not
widened**. Sweep EchoMuon's learning rate separately from Muon's: a grid that sits too
high for *both* arms flatters EchoMuon by roughly 0.046 in validation loss, which is
comparable to the margins being measured.

(An earlier version of this file claimed the optimum was consistently exactly 2×
Muon's. Our learning-rate grids are geometric with ratio 2, so any difference in the
discrete pick is necessarily "2×"; that number measured grid resolution rather than
the shift. The table above is the continuous measurement.)

## What the audit established

| Setting | Result | Status |
|---|---|---|
| Tiny ImageNet (clean + 20% label noise) | **+1.65 pp** vs lr-matched Muon (t=+7.4, n=8 paired) | Verified: the lr optimum is interior, and a norm-matched scalar control carrying the same average contraction reproduces only **30%** of the gain (EchoMuon beats it at t=+4.1), so the effect is **directional**, not step-size |
| CIFAR-10 / CIFAR-100 (4 cells) | +0.26 to +0.57 pp per cell; pooled **+0.35 pp** (95% CI +0.12 to +0.58) | **No single cell reaches significance**; the family-level effect does. Single-seed lr selection flipped one pick with a 1.45 pp consequence on test accuracy, which 3-seed selection corrected |
| FineWeb LM, 0.3 tok/param | −0.031 nats (t=−6.3) | Holds at this budget |
| FineWeb LM, 0.9 tok/param | −0.003 nats (t=−0.9) | **The LM advantage vanishes with token budget** |
| enwik8 bytes, clean, 38M | −0.0030 nats (t=−10.1) | Holds |
| enwik8 bytes, 10% corrupted | loses | Input-stream corruption is a separate, adverse regime |
| Mamba-2 SSM | ties Muon; AdamW beats the whole Muon family | Adverse |

**The margin follows a measured law.** Across ten vision cells spanning three
datasets, four label-noise levels, two budgets and two profiles, the advantage
scales with how far the model is from the quality it can reach:

```
margin (pp)  ~  0.027 x (baseline error rate %)  -  0.28
```

It crosses zero at roughly **90% accuracy**. Above that, expect nothing. The LM
budget ladder points the same way. This is the single most useful thing to know
before trying it.

## Install

```bash
pip install echomuon          # released package
# or, for the full experiment pipeline:
git clone https://github.com/MStamatis/echomuon && cd echomuon
```

## Usage

EchoMuon handles only the 2-D hidden matrices, exactly like Muon; route embeddings,
heads, gains and biases to AdamW:

```python
import torch
from echomuon import EchoMuon, MemorizationGapController

hidden = [p for n, p in model.named_parameters()
          if p.ndim == 2 and "embed" not in n and "lm_head" not in n]
others = [p for n, p in model.named_parameters()
          if not any(p is h for h in hidden)]

opt = EchoMuon(hidden, lr=0.02)   # sweep separately from Muon's; see note above
aux = torch.optim.AdamW(others, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
```

Training loop with the retention-gap controller (optional; without it, set
`gate_lambda` yourself — `gate_lambda=0` is plain Muon):

```python
import collections

ctl = MemorizationGapController(opt)     # probes every 200 steps by default
ring = collections.deque(maxlen=8)       # the last 8 training batches

for step, batch in enumerate(loader):
    loss = model(**batch).loss
    opt.zero_grad(); aux.zero_grad()
    loss.backward()
    opt.step(); aux.step()
    ring.append(batch)

    if ctl.due(step):
        with torch.no_grad():
            reseen = mean_loss(model, list(ring)[:4])   # ~4-8 steps old, see note
            fresh  = mean_loss(model, take_fresh(4))    # 4 held-back fresh batches
        ctl.update(fresh_loss=fresh, reseen_loss=reseen)
```

> **Note on the retention horizon.** Appending every step, as above, makes the
> re-seen batches only ~4–8 steps old — not the few hundred that earlier
> documentation claimed. We tested the intended long horizon (append once per probe,
> giving 400–700 steps) on five cells: **it changes nothing.** The controller is
> insensitive to this horizon over two orders of magnitude, so the short ring is fine
> — but the "memorization over hundreds of steps" story that motivated it is not
> supported.

All constants (slow β=0.99, floor 0.1, median reference, the 2% normalizer, the 0.7
EMA) were fixed once and used unchanged throughout. The knob you tune is the
learning rate — separately from Muon's.

## When it might help

| Your setting | Recommendation |
|---|---|
| Baseline accuracy well below ~90%, fixed step budget | Worth measuring — this is where any margin lives |
| Baseline near its ceiling, or strong augmentation | Expect a tie; the margin is gone by ~90% accuracy |
| LM pretraining at realistic token budgets | **Not recommended** — the advantage was gone by 0.9 tok/param, still only ~4.5% of compute-optimal |
| Byte-level LMs on corrupted input streams | **Scheduled Muon** — measured loss |
| Mamba-style SSMs | **AdamW** — beats the whole Muon family there |
| Multi-GPU / FSDP / tensor-parallel | **Not supported** — see Limitations |

## Limitations

- **Single-GPU only.** The gate needs the unsharded matrix to form its Gram product.
  FSDP or tensor-parallel execution would need an all-gather per refresh, or
  per-shard gates with different semantics. Untested.
- **Memory:** 9–12 B per 2-D parameter (M₁, M₂, cached basis U, gate g, all fp32),
  versus 4 B for Muon and 8 B for AdamW.
- **Wall-clock:** +19–22% per step at LM scale with the standard profile (median
  +21%), +13% with the fast profile. The vision overhead figures are unreliable:
  those models are small enough to be CPU-launch-bound rather than GPU-bound.
- **No head-to-head comparison** against other gated-Muon variants (DynMuon, Pion,
  MGUP, MAGMA, Bi-Maxwell). Everything here is measured against Muon and AdamW only.
  Any claim of superiority over those methods would be unsupported.
- **Learning-rate selection is single-seed on most cells.** Three-seed selection was
  used only where one seed proved insufficient; we demonstrated one case where a
  single-seed pick flipped with a 1.45 pp consequence on the test metric. Report your
  own selection margins.
- **Largest model tested: 162M parameters.** Nothing here has been tested at
  production scale.

## Paper

S. Mastromichalakis, *EchoMuon: Cross-Timescale Gating in Muon, and the Learning-Rate
Confound in Gated Optimizers*, Preprints.org, 2026.
doi:[10.20944/preprints202608.2150.v1](https://doi.org/10.20944/preprints202608.2150.v1)

The paper's figures and tables are generated directly from the run logs in this
repository by [`lab/make_figures.py`](lab/make_figures.py) and the report scripts
under `lab/`.

**Added after preprint v1, not in the paper.** A follow-up tested two proposed
repairs of the gate: an absolute noise-floor reference in place of the layer median,
and a controller driven by the gate's own signal fraction in place of the retention
gap. The diagnosis behind both held up; neither repair pays. The signal-fraction
controller gives up 0.57 pp of the Tiny ImageNet margin against the shipped gate
(t=−2.99, n=8) and still loses to Muon on corrupted bytes (t=+2.89, n=16). The
absolute reference shows no detectable gain on Tiny ImageNet (+0.44 pp, t=+0.70,
n=3) and widens the corrupted-byte loss by a further 0.0039 nats (t=+5.29, n=3).
Code and logs: [`lab/c_probe.py`](lab/c_probe.py),
[`lab/v3_report.py`](lab/v3_report.py), `lab/v3screen.log`, `lab/v3confirm.log`.
(`lab/v3_numbers.py`, `v3_tables.py` and `v3_figure.py` are unrelated despite the
name: they build the preprint's own tables.)

## Repository layout

```
src/echomuon/   the pip package: EchoMuon optimizer + MemorizationGapController
tests/          CPU test suite (python tests/test_echomuon.py)
lab/            the full experiment pipeline (single RTX 5090, Docker, step-resumable)
lab/results/    final.json + config.json for all 1,815 runs, including the audit
```

## The audit is part of the artifact

This repository contains not only the runs that support the method, but the runs that
attacked it — and won, in several places. Anyone re-examining a gated optimizer can
reuse the same harness:

| Check | Stage | What it caught here |
|---|---|---|
| Norm-matched scalar control | `v2-controls` | Separates directional signal from step-size contraction (30% vs 70% on Tiny ImageNet) |
| Widened lr grids | `v2-byte-grid`, `v2-valsplit` | 8/8 CIFAR arm-cells sat on a grid edge; correcting cut margins 42–80% |
| Held-out-val lr selection | `v2-valsplit` | Selection had been touching the test set |
| Split-robustness re-selection | `v2-resweep-cifar` | One pick flipped, with a 1.45 pp test consequence |
| Multi-seed lr selection | `v2-multiseed` | Quantifies selection noise against effect size |
| True-horizon retention ring | `v2-horizon` | The controller is horizon-insensitive; the original story was wrong |
| Budget-matched control | `v2-budget-match` | The margin tracks budget, not recipe |
| Token-budget ladder | `v2-fa3x` | The LM advantage vanishes by 0.9 tok/param |

If you maintain a gated-Muon variant, the first two rows are the cheapest way to find
out whether your reported margin is a mechanism or a learning-rate artifact.

## Reproducing

Every number is a function of `lab/results/runs/*/final.json` — the report scripts
(`lab/ablation_report.py`, `lab/v2_analysis.py`, `lab/fastprof_report.py`,
`lab/time_to_target.py`) and `lab/make_figures.py` recompute tables and figures from
those logs directly. To re-run training from scratch:

```bash
cd lab
docker build -t optlab .
docker run --gpus all -v $PWD:/lab optlab <stage>   # stages listed in run_experiments.py
```

Runs are step-level resumable; sweeps, finals, controls and audits are separate
stages with pre-registered predictions recorded in their docstrings.

The installable package and the research code are now pinned to each other by
`tests/test_parity.py`, which runs both on the same gradients and requires the updates to
match bit for bit. They did not before 0.3.0: the package refreshed the gate basis after
folding in the current gradient and `lab/src/optim.py` did so before, a one-step offset
that nothing tested, since `lab/v2_bitcheck.py` compares the lab against an older copy of
itself rather than against the package. On a controlled comparison the effect was smaller
than seed-to-seed noise and not statistically distinguishable at n=8, so no published
number changes, but a package that claims to reproduce a paper should reproduce it
exactly. **0.3.0 moves the package onto the paper's order.**

## Citation

```bibtex
@misc{mastromichalakis2026echomuon,
  title     = {EchoMuon: Cross-Timescale Gating in Muon, and the Learning-Rate
               Confound in Gated Optimizers},
  author    = {Mastromichalakis, Stamatis},
  year      = {2026},
  publisher = {Preprints.org},
  doi       = {10.20944/preprints202608.2150.v1},
  url       = {https://doi.org/10.20944/preprints202608.2150.v1},
  note      = {Preprint}
}
```

## License

MIT
