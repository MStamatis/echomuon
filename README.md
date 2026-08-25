# EchoMuon

**Muon with a per-direction temporal trust gate and a memorization-gap controller.
Better than scheduled Muon wherever data are imperfect; ties it everywhere else.**

[Muon](https://kellerjordan.github.io/posts/muon/) orthogonalizes the momentum of
2-D hidden weight matrices, giving every singular direction of the update exactly
equal trust. EchoMuon prices that trust by each direction's *echo* — its support in
a second, slower momentum buffer (β=0.99 next to Muon's 0.95). Persistent signal
appears in both buffers; noise flickers in the fast one and leaves no trace in the
slow one. The gate:

- scores each singular direction by cross-timescale **agreement**
  `c_i = uᵢᵀ M₂M₁ᵀ uᵢ / σᵢ²` — *not* by magnitude, so a large direction sustained by
  a few noisy batches is damped while a small persistent one is trusted;
- is **median-normalized per layer** (mean ≈ 1): trust is reallocated across
  directions at constant total step, so the gate cannot act as a disguised
  learning-rate schedule;
- is engaged in proportion to a **measured memorization gap** λ ∈ [0, 1]: the loss
  on fresh batches minus the loss on batches seen a few hundred steps ago, under
  the same current weights. At λ=0 EchoMuon *is* Muon, structurally — never worse
  by construction on clean data.

Headline results (paired seeds, schedule parity, per-arm lr sweeps): **+0.9 to
+1.8pp** over scheduled Muon on six vision cells (CIFAR-10/100, Tiny ImageNet,
clean and 20% label noise), **−0.031 nats (t=−6.3)** on a LLaMA-style 162M
transformer on FineWeb-Edu, Muon-tier quality in 80–85% of Muon's steps in every
seed, ~13% step overhead at 162M with the fast profile. See the paper for the
boundaries (byte-level LMs, SSMs, strong augmentation recipes) — they are reported,
measured, and part of the result.

## Install

```bash
pip install echomuon
```

## Usage

EchoMuon handles only the 2-D hidden matrices, exactly like Muon; route
embeddings, heads, gains and biases to AdamW:

```python
import torch
from echomuon import EchoMuon, MemorizationGapController

hidden = [p for n, p in model.named_parameters()
          if p.ndim == 2 and "embed" not in n and "lm_head" not in n]
others = [p for n, p in model.named_parameters()
          if not any(p is h for h in hidden)]

opt = EchoMuon(hidden, lr=0.02)                      # sweep lr as you would for Muon
aux = torch.optim.AdamW(others, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
```

Training loop with the memorization-gap controller (optional but recommended —
without it, set `gate_lambda` yourself; `gate_lambda=0` is plain Muon):

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
            reseen = mean_loss(model, list(ring)[:4])   # oldest ~400-800 steps ago
            fresh  = mean_loss(model, take_fresh(4))    # 4 held-back fresh batches
        ctl.update(fresh_loss=fresh, reseen_loss=reseen)
```

All constants (slow β=0.99, floor 0.1, median reference, the 2% normalizer, the
0.7 EMA) were fixed once and used unchanged in every experiment of the paper —
the only knob you tune is Muon's own learning rate.

## When to use it

| Your setting | Recommendation |
|---|---|
| Web-scale corpora, label noise, ambiguous labels, light augmentation | **EchoMuon** — this is where the margins live |
| Clean data / strong augmentation (RandAugment + mixup) | Tie with Muon; λ backs the gate off automatically |
| Byte-level LMs at scale | Scheduled Muon (measured boundary; EchoMuon concedes ≤0.5%) |
| Mamba-style SSMs | AdamW beats the whole Muon family there (measured boundary) |

## Citation

```bibtex
@article{mastromichalakis2026echomuon,
  title  = {EchoMuon: Better Than Scheduled Muon Wherever Data Are Imperfect},
  author = {Mastromichalakis, Stamatis},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## License

MIT

## Paper

arXiv link to follow. The paper's figures are generated directly from the run logs in
this repository by [`lab/make_figures.py`](lab/make_figures.py).

## Repository layout

```
src/echomuon/   the pip package: EchoMuon optimizer + MemorizationGapController
tests/          CPU test suite (python tests/test_echomuon.py)
lab/            the full experiment pipeline (single RTX 5090, Docker, step-resumable)
lab/results/    final.json + config.json for all 1250 runs behind the paper's numbers
```

## Reproducing the paper

Every number in the paper is a function of `lab/results/runs/*/final.json` — the report
scripts (`lab/ablation_report.py`, `lab/tiny_report.py`, `lab/phase2_report.py`,
`lab/time_to_target.py`, `lab/fastprof_report.py`) and `lab/make_figures.py` recompute
tables and figures from those logs directly. To re-run training from scratch:

```bash
cd lab
docker build -t optlab .
docker run --gpus all -v $PWD:/lab optlab <stage>   # stages listed in run_experiments.py
```

Runs are step-level resumable; sweeps, finals, and ablations are separate stages with
pre-registered predictions recorded in their docstrings.
