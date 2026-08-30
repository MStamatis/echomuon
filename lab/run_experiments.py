"""Experiment orchestrator. Stages:
  smoke    - 1 short run per optimizer, verifies the whole pipeline (~2-3 min on GPU)
  sweep    - lr grid x {regime A (clean), regime B (low SNR)} x 1 seed        [pretraining protocol]
  final    - best lr from sweep x 3 seeds, plus muon+spectral-lr controller   [pretraining protocol]
  pretrain - train the shared Muon base model on enwik8, save base checkpoint [fine-tuning protocol]
  ft-sweep - fine-tune lr grid x {F1 batch64, F2 batch8} x 1 seed             [fine-tuning protocol]
  ft-final - best lr x 3 seeds per arm                                        [fine-tuning protocol]
  analyze  - tables + plots -> results/report.md
  all      - smoke -> sweep -> final -> analyze
  ft-all   - pretrain -> ft-sweep -> ft-final -> analyze
Fully resumable: a run whose final.json exists is skipped, and an interrupted run
resumes from its last ckpt.pt (written every --ckpt-every steps) — safe across
crashes and PC restarts; just re-run the same command.
"""
import itertools
import json
import os
import subprocess
import sys

RESULTS = os.environ.get("RESULTS_DIR", "results")

REGIMES = {"A": 0.0, "B": 1.0}  # name -> grad_noise
GRIDS = {
    "adamw": [6e-4, 2e-3, 6e-3],
    "muon": [0.01, 0.02, 0.05, 0.1],
    "shrunk": [0.01, 0.02, 0.05, 0.1],
}
# enwik8 budgets: final = 4000 steps x 16k tokens = 65M tokens < 1 epoch of the 95M-token
# train split, so final val loss measures optimization quality, not overfitting speed
# (the v1 shakespeare protocol trained ~100 epochs and mostly measured memorization).
DATASET = "enwik8"
SWEEP_STEPS = 1500
FINAL_STEPS = 4000
FINAL_SEEDS = [1, 2, 3]


def run_one(run_id: str, **kw):
    cmd = [sys.executable, "-m", "src.train", "--run-id", run_id]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]
    print("=" * 80, flush=True)
    print("RUN:", run_id, flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"run {run_id} failed with code {r.returncode}")


def stage_smoke():
    for opt in ["adamw", "muon", "shrunk"]:
        run_one(f"smoke_{opt}", optimizer=opt, lr=GRIDS[opt][1], steps=200,
                eval_every=100, monitor_every=50, seed=1)
    run_one("smoke_shrunk_noise", optimizer="shrunk", lr=GRIDS["shrunk"][1], steps=200,
            eval_every=100, monitor_every=50, seed=1, grad_noise=1.0)
    print("\nSMOKE OK - all optimizers ran end to end.")


def stage_sweep():
    for regime, noise in REGIMES.items():
        for opt, lrs in GRIDS.items():
            for lr in lrs:
                run_one(f"sweep_{regime}_{opt}_lr{lr:g}", optimizer=opt, lr=lr,
                        dataset=DATASET, steps=SWEEP_STEPS, grad_noise=noise,
                        seed=1, no_monitor=True)


def best_lr(regime: str, opt: str):
    runs = []
    for lr in GRIDS[opt]:
        p = os.path.join(RESULTS, "runs", f"sweep_{regime}_{opt}_lr{lr:g}", "final.json")
        if os.path.exists(p):
            with open(p) as f:
                runs.append((json.load(f)["final_val"], lr))
    if not runs:
        raise RuntimeError(f"no sweep results for {regime}/{opt}; run the sweep stage first")
    runs.sort()
    val, lr = runs[0]
    if lr in (GRIDS[opt][0], GRIDS[opt][-1]):
        print(f"WARNING: best lr for {regime}/{opt} is at grid edge ({lr}); consider widening the grid")
    return lr


def stage_final():
    for regime, noise in REGIMES.items():
        for opt in GRIDS:
            lr = best_lr(regime, opt)
            for seed in FINAL_SEEDS:
                run_one(f"final_{regime}_{opt}_s{seed}", optimizer=opt, lr=lr,
                        dataset=DATASET, steps=FINAL_STEPS, grad_noise=noise, seed=seed)
        # Idea-3 controller arm on top of Muon at its best lr
        lr = best_lr(regime, "muon")
        for seed in FINAL_SEEDS:
            run_one(f"final_{regime}_muonctl_s{seed}", optimizer="muon", lr=lr,
                    dataset=DATASET, steps=FINAL_STEPS, grad_noise=noise, seed=seed,
                    spectral_lr=True)


# ---------------- Idea-3 controller follow-up: power + ablations ----------------
# Confirmation: muon vs muon+ctl at 8 seeds in both regimes (v2 seeds 1-3 are reused via
# final.json skip; run ids stay in the final_* namespace so analyze aggregates them).
# Ablations (regime B only, where the v2 effect lives): inverse controller must HURT if
# the stable-rank signal is directional; shuffled must lose the benefit if the
# layer-to-scale assignment (not just scale diversity) is what matters.
CTL_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]


def stage_ctl():
    for regime, noise in REGIMES.items():
        lr = best_lr(regime, "muon")
        for seed in CTL_SEEDS:
            run_one(f"final_{regime}_muon_s{seed}", optimizer="muon", lr=lr,
                    dataset=DATASET, steps=FINAL_STEPS, grad_noise=noise, seed=seed)
            run_one(f"final_{regime}_muonctl_s{seed}", optimizer="muon", lr=lr,
                    dataset=DATASET, steps=FINAL_STEPS, grad_noise=noise, seed=seed,
                    spectral_lr=True)
    lr = best_lr("B", "muon")
    for mode, tag in [("inverse", "muonctlinv"), ("shuffled", "muonctlshuf")]:
        for seed in CTL_SEEDS:
            run_one(f"final_B_{tag}_s{seed}", optimizer="muon", lr=lr,
                    dataset=DATASET, steps=FINAL_STEPS, grad_noise=REGIMES["B"], seed=seed,
                    spectral_lr=True, ctl_mode=mode)


# ---------------- Idea-3 scale test: does the ctl effect grow with model size? ----------------
# Same protocol as the v4 confirmation (enwik8, regime B grad-noise, muon vs muon+ctl at
# muon's best lr, fixed 4000-step token budget, head_dim 64 everywhere). Scale S (11M)
# is the existing final_B_{muon,muonctl} set; M and L get their own mini lr sweep first.
SCALES = {
    "M": {"n_layer": 12, "n_head": 8, "dim": 512, "grid": [0.01, 0.02, 0.04],
          "seeds": [1, 2, 3, 4, 5, 6, 7, 8]},
    "L": {"n_layer": 16, "n_head": 12, "dim": 768, "grid": [0.01, 0.02, 0.04],
          "seeds": [1, 2, 3, 4, 5, 6]},
}
SCALE_SWEEP_STEPS = 2000


def scale_dims(size):
    cfg = SCALES[size]
    return {"n_layer": cfg["n_layer"], "n_head": cfg["n_head"], "dim": cfg["dim"]}


def stage_scale_sweep():
    for size, cfg in SCALES.items():
        for lr in cfg["grid"]:
            run_one(f"scale_{size}_sweep_muon_lr{lr:g}", optimizer="muon", lr=lr,
                    dataset=DATASET, steps=SCALE_SWEEP_STEPS, grad_noise=REGIMES["B"],
                    seed=1, no_monitor=True, **scale_dims(size))


def scale_best_lr(size):
    runs = []
    for lr in SCALES[size]["grid"]:
        p = os.path.join(RESULTS, "runs", f"scale_{size}_sweep_muon_lr{lr:g}", "final.json")
        if os.path.exists(p):
            with open(p) as f:
                runs.append((json.load(f)["final_val"], lr))
    if not runs:
        raise RuntimeError(f"no scale sweep results for {size}; run scale-sweep first")
    runs.sort()
    val, lr = runs[0]
    if lr in (SCALES[size]["grid"][0], SCALES[size]["grid"][-1]):
        print(f"WARNING: best lr for scale {size} is at grid edge ({lr})")
    return lr


def stage_scale_final():
    for size, cfg in SCALES.items():
        lr = scale_best_lr(size)
        for seed in cfg["seeds"]:
            run_one(f"scale_{size}_muon_s{seed}", optimizer="muon", lr=lr, dataset=DATASET,
                    steps=FINAL_STEPS, grad_noise=REGIMES["B"], seed=seed, **scale_dims(size))
            run_one(f"scale_{size}_muonctl_s{seed}", optimizer="muon", lr=lr, dataset=DATASET,
                    steps=FINAL_STEPS, grad_noise=REGIMES["B"], seed=seed, spectral_lr=True,
                    **scale_dims(size))


# ---------------- spectral-ctl (πρώην SpectralMuon): the integrated controller (rank + conf + valve) ----------------
# Test matrix — each regime is one component's home turf:
#   A (clean pretrain)      must not hurt vs muon            [safety]
#   B (noisy pretrain)      must be >= muon+ctl              [rank + conf]
#   C (stress: 3x lr, no grad clip, clean) muon should spike/diverge; the valve
#     should survive; sm-novalve ablation attributes the effect   [valve]
#   D (fine-tune F2)        conf gate should shrink the overfit gap toward what
#     full ShrunkMuon achieved, at zero extra cost           [conf]
SM_SIGNALS = "rank,conf,valve"
SM_GRIDS = {"A": [0.01, 0.02, 0.04], "B": [0.02, 0.04, 0.08]}  # conf scaling lowers
SM_FT_GRID = [2e-3, 5e-3, 1e-2]                                # effective lr -> wider grids


def stage_sm_sweep():
    for regime, grid in SM_GRIDS.items():
        for lr in grid:
            run_one(f"sm_sweep_{regime}_spectralmuon_lr{lr:g}", optimizer="muon", lr=lr,
                    dataset=DATASET, steps=SWEEP_STEPS, grad_noise=REGIMES[regime], seed=1,
                    spectral_lr=True, ctl_signals=SM_SIGNALS)
    for lr in SM_FT_GRID:
        run_one(f"ft_sweep_F2_spectralmuon_lr{lr:g}", lr=lr, steps=FT_SWEEP_STEPS, seed=1,
                spectral_lr=True, ctl_signals=SM_SIGNALS, **ft_kwargs("muon", "F2"))


def sm_best_lr(regime):
    prefix = (f"sm_sweep_{regime}_spectralmuon_lr", SM_GRIDS[regime]) if regime in SM_GRIDS \
        else (f"ft_sweep_F2_spectralmuon_lr", SM_FT_GRID)
    runs = []
    for lr in prefix[1]:
        p = os.path.join(RESULTS, "runs", f"{prefix[0]}{lr:g}", "final.json")
        if os.path.exists(p):
            with open(p) as f:
                runs.append((json.load(f)["final_val"], lr))
    if not runs:
        raise RuntimeError(f"no spectral-ctl sweep results for {regime}; run sm-sweep first")
    runs.sort()
    if runs[0][1] in (prefix[1][0], prefix[1][-1]):
        print(f"WARNING: best spectral-ctl lr for {regime} is at grid edge ({runs[0][1]})")
    return runs[0][1]


def stage_sm_final():
    # A/B pretraining: muon and muon+ctl baselines already exist at these run ids
    for regime, seeds in [("A", [1, 2, 3, 4]), ("B", CTL_SEEDS)]:
        lr = sm_best_lr(regime)
        for seed in seeds:
            run_one(f"final_{regime}_spectralmuon_s{seed}", optimizer="muon", lr=lr,
                    dataset=DATASET, steps=FINAL_STEPS, grad_noise=REGIMES[regime], seed=seed,
                    spectral_lr=True, ctl_signals=SM_SIGNALS)
    # C stress: clean data, 3x muon's best clean lr, gradient clipping OFF
    lr_c = 3 * best_lr("A", "muon")
    for seed in CTL_SEEDS:
        run_one(f"final_C_muon_s{seed}", optimizer="muon", lr=lr_c, dataset=DATASET,
                steps=FINAL_STEPS, grad_noise=0.0, grad_clip=0.0, seed=seed)
        run_one(f"final_C_spectralmuon_s{seed}", optimizer="muon", lr=lr_c, dataset=DATASET,
                steps=FINAL_STEPS, grad_noise=0.0, grad_clip=0.0, seed=seed,
                spectral_lr=True, ctl_signals=SM_SIGNALS)
        run_one(f"final_C_smnovalve_s{seed}", optimizer="muon", lr=lr_c, dataset=DATASET,
                steps=FINAL_STEPS, grad_noise=0.0, grad_clip=0.0, seed=seed,
                spectral_lr=True, ctl_signals="rank,conf")
    # D fine-tuning: extend muon to 6 seeds, spectral-ctl at its own best ft lr
    lr_m = ft_best_lr("F2", "muon")
    for seed in [4, 5, 6]:
        run_one(f"ft_final_F2_muon_s{seed}", lr=lr_m, steps=FT_FINAL_STEPS, seed=seed,
                **ft_kwargs("muon", "F2"))
    lr_sm = sm_best_lr("F2")
    for seed in [1, 2, 3, 4, 5, 6]:
        run_one(f"ft_final_F2_spectralmuon_s{seed}", lr=lr_sm, steps=FT_FINAL_STEPS, seed=seed,
                spectral_lr=True, ctl_signals=SM_SIGNALS, **ft_kwargs("muon", "F2"))


# ---------------- spectral-ctl v2 (πρώην SpectralMuon): trend-gated conf + valve fault-injection test ----------------
# v1 verdict: D win (conf captured full-shrinkage benefit at zero cost), A safe,
# C revealed lr-robustness, but B regressed (always-on conf gates healthy noisy
# learning) and the valve never fired. v2: conf2 gates on w-bar's DECLINE from its
# running max instead of its absolute level; the valve gets a controlled fault
# (20x lr for 50 steps mid-training, no grad clipping) it must detect and contain.
SM2_SIGNALS = "rank,conf2,valve"


def _pick_best(prefix, lrs):
    runs = []
    for lr in lrs:
        p = os.path.join(RESULTS, "runs", f"{prefix}{lr:g}", "final.json")
        if os.path.exists(p):
            with open(p) as f:
                runs.append((json.load(f)["final_val"], lr))
    if not runs:
        raise RuntimeError(f"no sweep results at {prefix}*")
    runs.sort()
    return runs[0][1]


def stage_sm2_clean():
    """The final config (rank,conf2,valve) on canonical clean pretraining (regime A) —
    the safety row of the claims matrix, previously covered only by the v1 config at n=4."""
    for lr in [0.01, 0.02, 0.04]:
        run_one(f"sm_sweep_A_sm2_lr{lr:g}", optimizer="muon", lr=lr, dataset=DATASET,
                steps=SWEEP_STEPS, grad_noise=0.0, seed=1,
                spectral_lr=True, ctl_signals=SM2_SIGNALS)
    lr = _pick_best("sm_sweep_A_sm2_lr", [0.01, 0.02, 0.04])
    for seed in CTL_SEEDS:
        run_one(f"final_A_sm2_s{seed}", optimizer="muon", lr=lr, dataset=DATASET,
                steps=FINAL_STEPS, grad_noise=0.0, seed=seed,
                spectral_lr=True, ctl_signals=SM2_SIGNALS)


def stage_cos_control():
    """Control for sm2's clean-regime win: is it just an implicit lr schedule?
    Muon + cosine decay on the matrix lr, same protocol as regime A finals."""
    for lr in [0.01, 0.02, 0.04]:
        run_one(f"sweep_A_muoncos_lr{lr:g}", optimizer="muon", lr=lr, dataset=DATASET,
                steps=SWEEP_STEPS, grad_noise=0.0, seed=1, no_monitor=True,
                lr_schedule="cosine")
    lr = _pick_best("sweep_A_muoncos_lr", [0.01, 0.02, 0.04])
    for seed in CTL_SEEDS:
        run_one(f"final_A_muoncos_s{seed}", optimizer="muon", lr=lr, dataset=DATASET,
                steps=FINAL_STEPS, grad_noise=0.0, seed=seed, lr_schedule="cosine")


def stage_sched_parity():
    """The decisive practicality test: do the natural-noise wins survive when BOTH arms
    get a cosine schedule (standard practice)? Runs the two strongest noise cells."""
    # P: corrupted-data LM
    for arm, extra in [("muoncos", {}), ("sm2cos", {"spectral_lr": True,
                                                    "ctl_signals": SM2_SIGNALS})]:
        for lr in [0.01, 0.02, 0.04]:
            run_one(f"sweep_P_{arm}_lr{lr:g}", optimizer="muon", lr=lr, dataset="enwik8p10",
                    batch=64, steps=2000, seed=1, lr_schedule="cosine",
                    no_monitor=not extra, **extra)
        lr = _pick_best(f"sweep_P_{arm}_lr", [0.01, 0.02, 0.04])
        for seed in CTL_SEEDS:
            run_one(f"final_P_{arm}_s{seed}", optimizer="muon", lr=lr, dataset="enwik8p10",
                    batch=64, steps=4000, seed=seed, lr_schedule="cosine", **extra)
    # VP: label-noise vision
    for arm, extra in [("muoncos", {}), ("sm2cos", {"spectral_lr": True,
                                                    "ctl_signals": SM2_SIGNALS})]:
        for lr in V_GRID:
            run_one(f"sweep_VP_{arm}_lr{lr:g}", optimizer="muon", lr=lr, task="vision",
                    dataset="cifar10n20", batch=128, steps=V_SWEEP_STEPS, seed=1,
                    lr_schedule="cosine", no_monitor=not extra, **V_DIMS, **extra)
        lr = _pick_best(f"sweep_VP_{arm}_lr", V_GRID)
        for seed in CTL_SEEDS:
            run_one(f"final_VP_{arm}_s{seed}", optimizer="muon", lr=lr, task="vision",
                    dataset="cifar10n20", batch=128, steps=V_FINAL_STEPS, seed=seed,
                    lr_schedule="cosine", **V_DIMS, **extra)


def stage_tcg_pilot():
    """TCG (Temporal-Consistency Gating) pilot — protocol v2 lessons baked in:
    cosine schedule on EVERY arm, muon+cos baselines reused from sched-parity runs.
    Pre-registered predictions: beats muon+cos in P and VP; ties in clean A."""
    # P: corrupted-data LM (decisive — sm2 failed here under schedule parity)
    for lr in [0.01, 0.02, 0.04]:
        run_one(f"sweep_P_tcgcos_lr{lr:g}", optimizer="tcg", lr=lr, dataset="enwik8p10",
                batch=64, steps=2000, seed=1, lr_schedule="cosine", no_monitor=True)
    lr = _pick_best("sweep_P_tcgcos_lr", [0.01, 0.02, 0.04])
    for seed in CTL_SEEDS:
        run_one(f"final_P_tcgcos_s{seed}", optimizer="tcg", lr=lr, dataset="enwik8p10",
                batch=64, steps=4000, seed=seed, lr_schedule="cosine", no_monitor=True)
    # VP: label-noise vision (sm2 survived here — TCG must at least match)
    for lr in V_GRID:
        run_one(f"sweep_VP_tcgcos_lr{lr:g}", optimizer="tcg", lr=lr, task="vision",
                dataset="cifar10n20", batch=128, steps=V_SWEEP_STEPS, seed=1,
                lr_schedule="cosine", no_monitor=True, **V_DIMS)
    lr = _pick_best("sweep_VP_tcgcos_lr", V_GRID)
    for seed in CTL_SEEDS:
        run_one(f"final_VP_tcgcos_s{seed}", optimizer="tcg", lr=lr, task="vision",
                dataset="cifar10n20", batch=128, steps=V_FINAL_STEPS, seed=seed,
                lr_schedule="cosine", no_monitor=True, **V_DIMS)
    # A: clean safety (median-normalized gate should be a no-op on consistent signal)
    for lr in [0.01, 0.02, 0.04]:
        run_one(f"sweep_A_tcgcos_lr{lr:g}", optimizer="tcg", lr=lr, dataset=DATASET,
                batch=64, steps=2000, seed=1, lr_schedule="cosine", no_monitor=True)
    lr = _pick_best("sweep_A_tcgcos_lr", [0.01, 0.02, 0.04])
    for seed in [1, 2, 3, 4]:
        run_one(f"final_A_tcgcos_s{seed}", optimizer="tcg", lr=lr, dataset=DATASET,
                batch=64, steps=FINAL_STEPS, seed=seed, lr_schedule="cosine",
                no_monitor=True)


def stage_tcg_validate():
    """Causal ablations (inverse must hurt, shuffled must lose the effect) on P and VP,
    clean-vision safety (VA), and extending clean-LM A to 8 seeds. All arms cosine."""
    lr_p = _pick_best("sweep_P_tcgcos_lr", [0.01, 0.02, 0.04])
    lr_vp = _pick_best("sweep_VP_tcgcos_lr", V_GRID)
    for mode, tag in [("inverse", "tcginv"), ("shuffled", "tcgshuf")]:
        for seed in CTL_SEEDS:
            run_one(f"final_P_{tag}_s{seed}", optimizer="tcg", lr=lr_p, dataset="enwik8p10",
                    batch=64, steps=4000, seed=seed, lr_schedule="cosine",
                    gate_mode=mode, no_monitor=True)
            run_one(f"final_VP_{tag}_s{seed}", optimizer="tcg", lr=lr_vp, task="vision",
                    dataset="cifar10n20", batch=128, steps=V_FINAL_STEPS, seed=seed,
                    lr_schedule="cosine", gate_mode=mode, no_monitor=True, **V_DIMS)
    # VA clean-vision safety for TCG + its muon+cos baseline
    for arm, extra in [("muoncos", {}), ("tcgcos", {"optimizer": "tcg"})]:
        for lr in V_GRID:
            run_one(f"sweep_VA_{arm}_lr{lr:g}", optimizer=extra.get("optimizer", "muon"),
                    lr=lr, task="vision", dataset="cifar10", batch=128,
                    steps=V_SWEEP_STEPS, seed=1, lr_schedule="cosine",
                    no_monitor=True, **V_DIMS)
        lr = _pick_best(f"sweep_VA_{arm}_lr", V_GRID)
        for seed in CTL_SEEDS:
            run_one(f"final_VA_{arm}_s{seed}", optimizer=extra.get("optimizer", "muon"),
                    lr=lr, task="vision", dataset="cifar10", batch=128,
                    steps=V_FINAL_STEPS, seed=seed, lr_schedule="cosine",
                    no_monitor=True, **V_DIMS)
    # extend clean-LM A to 8 seeds
    lr_a = _pick_best("sweep_A_tcgcos_lr", [0.01, 0.02, 0.04])
    for seed in [5, 6, 7, 8]:
        run_one(f"final_A_tcgcos_s{seed}", optimizer="tcg", lr=lr_a, dataset=DATASET,
                batch=64, steps=FINAL_STEPS, seed=seed, lr_schedule="cosine",
                no_monitor=True)


def stage_tcg_scale():
    """TCG scale test on the corrupted-LM regime (P) at M-38M and L-114M, schedule
    parity, paired seeds — does the natural-noise win persist with model size?
    Plus the VA no-augmentation control for the clean-vision surprise."""
    scale_specs = [("M", {"n_layer": 12, "n_head": 8, "dim": 512}, [5e-3, 1e-2, 2e-2],
                    [1, 2, 3, 4, 5, 6, 7, 8]),
                   ("L", {"n_layer": 16, "n_head": 12, "dim": 768}, [5e-3, 1e-2, 2e-2],
                    [1, 2, 3, 4, 5, 6])]
    for size, dims, grid, seeds in scale_specs:
        for arm, opt in [("muoncos", "muon"), ("tcgcos", "tcg")]:
            for lr in grid:
                run_one(f"sweep_Pscale{size}_{arm}_lr{lr:g}", optimizer=opt, lr=lr,
                        dataset="enwik8p10", batch=64, steps=2000, seed=1,
                        lr_schedule="cosine", no_monitor=True, **dims)
            lr = _pick_best(f"sweep_Pscale{size}_{arm}_lr", grid)
            for seed in seeds:
                run_one(f"final_Pscale{size}_{arm}_s{seed}", optimizer=opt, lr=lr,
                        dataset="enwik8p10", batch=64, steps=4000, seed=seed,
                        lr_schedule="cosine", no_monitor=True, **dims)
    # VA without augmentation: if the clean-vision win came from augmentation
    # inconsistency, it should shrink here
    for arm, opt in [("muoncos", "muon"), ("tcgcos", "tcg")]:
        lr = _pick_best(f"sweep_VA_{arm}_lr", V_GRID)
        for seed in CTL_SEEDS:
            run_one(f"final_VAnoaug_{arm}_s{seed}", optimizer=opt, lr=lr, task="vision",
                    dataset="cifar10", batch=128, steps=V_FINAL_STEPS, seed=seed,
                    lr_schedule="cosine", no_augment=True, no_monitor=True, **V_DIMS)


def stage_tcg_q20():
    """TCG with a conservative gate (damp only the bottom-quintile consistency
    directions). One config across all four cells: P at S and M scale (the inversion
    point), plus VA/VP vision — the paper needs a single unified configuration."""
    q = {"gate_quantile": 0.2}
    lr_p = _pick_best("sweep_P_tcgcos_lr", [0.01, 0.02, 0.04])
    for seed in CTL_SEEDS:
        run_one(f"final_P_tcgq20_s{seed}", optimizer="tcg", lr=lr_p, dataset="enwik8p10",
                batch=64, steps=4000, seed=seed, lr_schedule="cosine",
                no_monitor=True, **q)
    lr_m = _pick_best("sweep_PscaleM_tcgcos_lr", [5e-3, 1e-2, 2e-2])
    for seed in CTL_SEEDS:
        run_one(f"final_PscaleM_tcgq20_s{seed}", optimizer="tcg", lr=lr_m,
                dataset="enwik8p10", batch=64, steps=4000, seed=seed,
                lr_schedule="cosine", no_monitor=True,
                n_layer=12, n_head=8, dim=512, **q)
    for reg, ds in [("VA", "cifar10"), ("VP", "cifar10n20")]:
        lr = _pick_best(f"sweep_{reg}_tcgcos_lr", V_GRID)
        for seed in CTL_SEEDS:
            run_one(f"final_{reg}_tcgq20_s{seed}", optimizer="tcg", lr=lr, task="vision",
                    dataset=ds, batch=128, steps=V_FINAL_STEPS, seed=seed,
                    lr_schedule="cosine", no_monitor=True, **V_DIMS, **q)


def stage_tcg_mfix():
    """Three diagnosed fixes for the LM scale inversion, tested at M-38M (the cheap
    inversion point): H3 pre-NS gating, H1 block-aggregated consistency, and the H2
    horizon diagnostic (does the harm shrink with longer training?)."""
    dims = {"n_layer": 12, "n_head": 8, "dim": 512}
    lr_m = _pick_best("sweep_PscaleM_tcgcos_lr", [5e-3, 1e-2, 2e-2])
    lr_mm = _pick_best("sweep_PscaleM_muoncos_lr", [5e-3, 1e-2, 2e-2])
    for seed in CTL_SEEDS:
        run_one(f"final_PscaleM_tcgpre_s{seed}", optimizer="tcg", lr=lr_m,
                dataset="enwik8p10", batch=64, steps=4000, seed=seed,
                lr_schedule="cosine", gate_stage="pre", no_monitor=True, **dims)
        run_one(f"final_PscaleM_tcgblk_s{seed}", optimizer="tcg", lr=lr_m,
                dataset="enwik8p10", batch=64, steps=4000, seed=seed,
                lr_schedule="cosine", gate_block=32, no_monitor=True, **dims)
    for seed in [1, 2, 3, 4]:
        run_one(f"final_PscaleMlong_muoncos_s{seed}", optimizer="muon", lr=lr_mm,
                dataset="enwik8p10", batch=64, steps=8000, seed=seed,
                lr_schedule="cosine", no_monitor=True, **dims)
        run_one(f"final_PscaleMlong_tcgcos_s{seed}", optimizer="tcg", lr=lr_m,
                dataset="enwik8p10", batch=64, steps=8000, seed=seed,
                lr_schedule="cosine", no_monitor=True, **dims)


def stage_tcg_beta():
    """H2 follow-up: longer slow-buffer memory (beta2 0.995 / 0.999) at M-38M with the
    8000-step horizon — if language's rare-feature tail is the culprit, more memory
    should erase the harm. Paired against the existing PscaleMlong muoncos runs."""
    dims = {"n_layer": 12, "n_head": 8, "dim": 512}
    lr_m = _pick_best("sweep_PscaleM_tcgcos_lr", [5e-3, 1e-2, 2e-2])
    for beta, tag in [(0.995, "tcgb995"), (0.999, "tcgb999")]:
        for seed in [1, 2, 3, 4]:
            run_one(f"final_PscaleMlong_{tag}_s{seed}", optimizer="tcg", lr=lr_m,
                    dataset="enwik8p10", batch=64, steps=8000, seed=seed,
                    lr_schedule="cosine", slow_beta=beta, no_monitor=True, **dims)


def stage_tcg_novelty():
    """Novelty-aware gate (three buffers: damp only low-support AND non-growing
    directions) at the two decisive LM cells: S-11M (must keep the win) and M-38M
    (must erase the inversion)."""
    q = {"gate_mode": "novelty", "gate_quantile": 0.2}
    lr_s = _pick_best("sweep_P_tcgcos_lr", [0.01, 0.02, 0.04])
    for seed in CTL_SEEDS:
        run_one(f"final_P_tcgnov_s{seed}", optimizer="tcg", lr=lr_s, dataset="enwik8p10",
                batch=64, steps=4000, seed=seed, lr_schedule="cosine",
                no_monitor=True, **q)
    lr_m = _pick_best("sweep_PscaleM_tcgcos_lr", [5e-3, 1e-2, 2e-2])
    for seed in CTL_SEEDS:
        run_one(f"final_PscaleM_tcgnov_s{seed}", optimizer="tcg", lr=lr_m,
                dataset="enwik8p10", batch=64, steps=4000, seed=seed,
                lr_schedule="cosine", no_monitor=True,
                n_layer=12, n_head=8, dim=512, **q)


def stage_ccg_pilot():
    """CCG (Conditional-Coherence Gating) pilot. Priority order: the M-38M graveyard
    cell first (six prior mechanisms died there), then S, then vision VP. Own lr sweeps,
    schedule parity, paired seeds vs existing muoncos baselines."""
    cc = {"gate_mode": "coherence"}
    for lr in [5e-3, 1e-2, 2e-2]:
        run_one(f"sweep_PscaleM_ccg_lr{lr:g}", optimizer="tcg", lr=lr, dataset="enwik8p10",
                batch=64, steps=2000, seed=1, lr_schedule="cosine", no_monitor=True,
                n_layer=12, n_head=8, dim=512, **cc)
    lr = _pick_best("sweep_PscaleM_ccg_lr", [5e-3, 1e-2, 2e-2])
    for seed in CTL_SEEDS:
        run_one(f"final_PscaleM_ccg_s{seed}", optimizer="tcg", lr=lr, dataset="enwik8p10",
                batch=64, steps=4000, seed=seed, lr_schedule="cosine", no_monitor=True,
                n_layer=12, n_head=8, dim=512, **cc)
    for lr in [0.01, 0.02, 0.04]:
        run_one(f"sweep_P_ccg_lr{lr:g}", optimizer="tcg", lr=lr, dataset="enwik8p10",
                batch=64, steps=2000, seed=1, lr_schedule="cosine", no_monitor=True, **cc)
    lr = _pick_best("sweep_P_ccg_lr", [0.01, 0.02, 0.04])
    for seed in CTL_SEEDS:
        run_one(f"final_P_ccg_s{seed}", optimizer="tcg", lr=lr, dataset="enwik8p10",
                batch=64, steps=4000, seed=seed, lr_schedule="cosine", no_monitor=True, **cc)
    for lr in V_GRID:
        run_one(f"sweep_VP_ccg_lr{lr:g}", optimizer="tcg", lr=lr, task="vision",
                dataset="cifar10n20", batch=128, steps=V_SWEEP_STEPS, seed=1,
                lr_schedule="cosine", no_monitor=True, **V_DIMS, **cc)
    lr = _pick_best("sweep_VP_ccg_lr", V_GRID)
    for seed in CTL_SEEDS:
        run_one(f"final_VP_ccg_s{seed}", optimizer="tcg", lr=lr, task="vision",
                dataset="cifar10n20", batch=128, steps=V_FINAL_STEPS, seed=seed,
                lr_schedule="cosine", no_monitor=True, **V_DIMS, **cc)


def stage_amp_pilot():
    """The last untried polarity: bounded amplification of coherent directions
    (mean-normalized), never suppression. M graveyard first, then S, then VP."""
    am = {"gate_mode": "amplify"}
    for lr in [5e-3, 1e-2, 2e-2]:
        run_one(f"sweep_PscaleM_amp_lr{lr:g}", optimizer="tcg", lr=lr, dataset="enwik8p10",
                batch=64, steps=2000, seed=1, lr_schedule="cosine", no_monitor=True,
                n_layer=12, n_head=8, dim=512, **am)
    lr = _pick_best("sweep_PscaleM_amp_lr", [5e-3, 1e-2, 2e-2])
    for seed in CTL_SEEDS:
        run_one(f"final_PscaleM_amp_s{seed}", optimizer="tcg", lr=lr, dataset="enwik8p10",
                batch=64, steps=4000, seed=seed, lr_schedule="cosine", no_monitor=True,
                n_layer=12, n_head=8, dim=512, **am)
    for lr in [0.01, 0.02, 0.04]:
        run_one(f"sweep_P_amp_lr{lr:g}", optimizer="tcg", lr=lr, dataset="enwik8p10",
                batch=64, steps=2000, seed=1, lr_schedule="cosine", no_monitor=True, **am)
    lr = _pick_best("sweep_P_amp_lr", [0.01, 0.02, 0.04])
    for seed in CTL_SEEDS:
        run_one(f"final_P_amp_s{seed}", optimizer="tcg", lr=lr, dataset="enwik8p10",
                batch=64, steps=4000, seed=seed, lr_schedule="cosine", no_monitor=True, **am)
    for lr in V_GRID:
        run_one(f"sweep_VP_amp_lr{lr:g}", optimizer="tcg", lr=lr, task="vision",
                dataset="cifar10n20", batch=128, steps=V_SWEEP_STEPS, seed=1,
                lr_schedule="cosine", no_monitor=True, **V_DIMS, **am)
    lr = _pick_best("sweep_VP_amp_lr", V_GRID)
    for seed in CTL_SEEDS:
        run_one(f"final_VP_amp_s{seed}", optimizer="tcg", lr=lr, task="vision",
                dataset="cifar10n20", batch=128, steps=V_FINAL_STEPS, seed=seed,
                lr_schedule="cosine", no_monitor=True, **V_DIMS, **am)


def stage_auto2_pilot():
    """EchoMuon v2: memorization-gap controller (fresh vs re-evaluated recently-seen
    batches). Same four cells; the timeboxed last controller iteration."""
    au = {"gate_mode": "normal", "auto_gate": True, "auto_version": 2}
    cells = [
        ("PscaleM", {"dataset": "enwik8p10", "batch": 64, "steps": 4000,
                     "n_layer": 12, "n_head": 8, "dim": 512},
         [5e-3, 1e-2, 2e-2], 2000),
        ("P", {"dataset": "enwik8p10", "batch": 64, "steps": 4000}, [0.01, 0.02, 0.04], 2000),
        ("VP", {"task": "vision", "dataset": "cifar10n20", "batch": 128,
                "steps": V_FINAL_STEPS, **V_DIMS}, V_GRID, V_SWEEP_STEPS),
        ("VA", {"task": "vision", "dataset": "cifar10", "batch": 128,
                "steps": V_FINAL_STEPS, **V_DIMS}, V_GRID, V_SWEEP_STEPS),
    ]
    for cell, cfg, grid, sweep_steps in cells:
        steps = cfg.pop("steps")
        for lr in grid:
            run_one(f"sweep_{cell}_auto2_lr{lr:g}", optimizer="tcg", lr=lr,
                    steps=sweep_steps, seed=1, lr_schedule="cosine",
                    no_monitor=True, **cfg, **au)
        lr = _pick_best(f"sweep_{cell}_auto2_lr", grid)
        for seed in CTL_SEEDS:
            run_one(f"final_{cell}_auto2_s{seed}", optimizer="tcg", lr=lr, steps=steps,
                    seed=seed, lr_schedule="cosine", no_monitor=True, **cfg, **au)


def stage_auto2_confirm():
    """Final characterization of EchoMuon v2: pin the M residual at n=16 (both arms)
    and measure the L-114M cell. Measurement only — no new mechanisms."""
    au = {"gate_mode": "normal", "auto_gate": True, "auto_version": 2}
    dims_m = {"n_layer": 12, "n_head": 8, "dim": 512}
    dims_l = {"n_layer": 16, "n_head": 12, "dim": 768}
    lr_am = _pick_best("sweep_PscaleM_auto2_lr", [5e-3, 1e-2, 2e-2])
    lr_mm = _pick_best("sweep_PscaleM_muoncos_lr", [5e-3, 1e-2, 2e-2])
    for seed in range(9, 17):
        run_one(f"final_PscaleM_auto2_s{seed}", optimizer="tcg", lr=lr_am,
                dataset="enwik8p10", batch=64, steps=4000, seed=seed,
                lr_schedule="cosine", no_monitor=True, **dims_m, **au)
        run_one(f"final_PscaleM_muoncos_s{seed}", optimizer="muon", lr=lr_mm,
                dataset="enwik8p10", batch=64, steps=4000, seed=seed,
                lr_schedule="cosine", no_monitor=True, **dims_m)
    for lr in [5e-3, 1e-2, 2e-2]:
        run_one(f"sweep_PscaleL_auto2_lr{lr:g}", optimizer="tcg", lr=lr,
                dataset="enwik8p10", batch=64, steps=2000, seed=1,
                lr_schedule="cosine", no_monitor=True, **dims_l, **au)
    lr_al = _pick_best("sweep_PscaleL_auto2_lr", [5e-3, 1e-2, 2e-2])
    for seed in [1, 2, 3, 4, 5, 6]:
        run_one(f"final_PscaleL_auto2_s{seed}", optimizer="tcg", lr=lr_al,
                dataset="enwik8p10", batch=64, steps=4000, seed=seed,
                lr_schedule="cosine", no_monitor=True, **dims_l, **au)


def stage_adamw_baselines():
    """The missing industry-standard baseline: AdamW + cosine at the four key cells.
    Determines whether EchoMuon's honest claim is 'beats AdamW everywhere, loses only
    to specialized scheduled Muon on large LMs by <=0.5%'."""
    for lr in [6e-4, 2e-3, 6e-3]:
        run_one(f"sweep_PscaleM_adamwcos_lr{lr:g}", optimizer="adamw", lr=lr,
                dataset="enwik8p10", batch=64, steps=2000, seed=1, lr_schedule="cosine",
                no_monitor=True, n_layer=12, n_head=8, dim=512)
    lr = _pick_best("sweep_PscaleM_adamwcos_lr", [6e-4, 2e-3, 6e-3])
    for seed in CTL_SEEDS:
        run_one(f"final_PscaleM_adamwcos_s{seed}", optimizer="adamw", lr=lr,
                dataset="enwik8p10", batch=64, steps=4000, seed=seed,
                lr_schedule="cosine", no_monitor=True, n_layer=12, n_head=8, dim=512)
    for lr in [6e-4, 2e-3]:
        run_one(f"sweep_PscaleL_adamwcos_lr{lr:g}", optimizer="adamw", lr=lr,
                dataset="enwik8p10", batch=64, steps=2000, seed=1, lr_schedule="cosine",
                no_monitor=True, n_layer=16, n_head=12, dim=768)
    lr = _pick_best("sweep_PscaleL_adamwcos_lr", [6e-4, 2e-3])
    for seed in [1, 2, 3, 4]:
        run_one(f"final_PscaleL_adamwcos_s{seed}", optimizer="adamw", lr=lr,
                dataset="enwik8p10", batch=64, steps=4000, seed=seed,
                lr_schedule="cosine", no_monitor=True, n_layer=16, n_head=12, dim=768)
    for reg, ds in [("VA", "cifar10"), ("VP", "cifar10n20")]:
        for lr in [3e-4, 1e-3, 3e-3]:
            run_one(f"sweep_{reg}_adamwcos_lr{lr:g}", optimizer="adamw", lr=lr,
                    task="vision", dataset=ds, batch=128, steps=V_SWEEP_STEPS, seed=1,
                    lr_schedule="cosine", no_monitor=True, **V_DIMS)
        lr = _pick_best(f"sweep_{reg}_adamwcos_lr", [3e-4, 1e-3, 3e-3])
        for seed in CTL_SEEDS:
            run_one(f"final_{reg}_adamwcos_s{seed}", optimizer="adamw", lr=lr,
                    task="vision", dataset=ds, batch=128, steps=V_FINAL_STEPS, seed=seed,
                    lr_schedule="cosine", no_monitor=True, **V_DIMS)


def stage_vision_breadth():
    """Phase 1 of the breadth expansion: CIFAR-100 (clean + 20% label noise), three
    arms (muon+cos, EchoMuon, adamw+cos), own sweeps, 8 paired seeds."""
    arms = [
        ("muoncos", {"optimizer": "muon"}, V_GRID),
        ("auto2", {"optimizer": "tcg", "gate_mode": "normal", "auto_gate": True,
                   "auto_version": 2}, V_GRID),
        ("adamwcos", {"optimizer": "adamw"}, [3e-4, 1e-3, 3e-3]),
    ]
    for reg, ds in [("V100A", "cifar100"), ("V100P", "cifar100n20")]:
        for arm, extra, grid in arms:
            for lr in grid:
                run_one(f"sweep_{reg}_{arm}_lr{lr:g}", lr=lr, task="vision", dataset=ds,
                        batch=128, steps=V_SWEEP_STEPS, seed=1, lr_schedule="cosine",
                        no_monitor=True, **V_DIMS, **extra)
            lr = _pick_best(f"sweep_{reg}_{arm}_lr", grid)
            for seed in CTL_SEEDS:
                run_one(f"final_{reg}_{arm}_s{seed}", lr=lr, task="vision", dataset=ds,
                        batch=128, steps=V_FINAL_STEPS, seed=seed, lr_schedule="cosine",
                        no_monitor=True, **V_DIMS, **extra)


def stage_vision_tiny():
    """Phase 1b (ready to fire after the CIFAR-100 report): Tiny-ImageNet-200 (64x64,
    200 classes) clean + 20% label noise, three arms, own sweeps, 8 paired seeds."""
    arms = [
        ("muoncos", {"optimizer": "muon"}, V_GRID),
        ("auto2", {"optimizer": "tcg", "gate_mode": "normal", "auto_gate": True,
                   "auto_version": 2}, V_GRID),
        ("adamwcos", {"optimizer": "adamw"}, [3e-4, 1e-3, 3e-3]),
    ]
    for reg, ds in [("TIA", "tinyimagenet"), ("TIP", "tinyimagenetn20")]:
        for arm, extra, grid in arms:
            for lr in grid:
                run_one(f"sweep_{reg}_{arm}_lr{lr:g}", lr=lr, task="vision", dataset=ds,
                        batch=128, steps=V_SWEEP_STEPS, seed=1, lr_schedule="cosine",
                        no_monitor=True, **V_DIMS, **extra)
            lr = _pick_best(f"sweep_{reg}_{arm}_lr", grid)
            for seed in CTL_SEEDS:
                run_one(f"final_{reg}_{arm}_s{seed}", lr=lr, task="vision", dataset=ds,
                        batch=128, steps=V_FINAL_STEPS, seed=seed, lr_schedule="cosine",
                        no_monitor=True, **V_DIMS, **extra)


def stage_lm_breadth():
    """Phase 2 (LM breadth): the modern-standard optimizer benchmark — FineWeb-Edu with
    GPT-2 BPE (the modded-nanogpt data) on two architectures:
      FA: LLaMA-style transformer (RMSNorm/RoPE/SwiGLU), 12L x 768d, 124M-class
      FM: Mamba-2-style SSM, 20L x 512d — the no-attention generalization cell
    Three arms with own sweeps, cosine everywhere, 6 paired seeds, ~49M tokens/run
    (<1 epoch of the 400M-token pool). Pre-registered predictions from the CIFAR/enwik8
    theory: muoncos <= auto2 (concession <=0.5% rel.) << adamwcos on FA; FM is open —
    it tests whether the flat-spectrum story is attention-specific."""
    arms = [
        ("muoncos", {"optimizer": "muon"}, [5e-3, 1e-2, 2e-2]),
        ("auto2", {"optimizer": "tcg", "gate_mode": "normal", "auto_gate": True,
                   "auto_version": 2}, [5e-3, 1e-2, 2e-2]),
        ("adamwcos", {"optimizer": "adamw"}, [3e-4, 6e-4, 1.2e-3]),
    ]
    cells = [
        ("FA", {"arch": "llama", "dataset": "fineweb", "batch": 16, "block": 1024,
                "n_layer": 12, "n_head": 12, "dim": 768}, 1250, 3000),
        ("FM", {"arch": "mamba", "dataset": "fineweb", "batch": 8, "block": 1024,
                "n_layer": 20, "dim": 512}, 2500, 6000),
    ]
    for reg, cfg, sweep_steps, final_steps in cells:
        for arm, extra, grid in arms:
            for lr in grid:
                run_one(f"sweep_{reg}_{arm}_lr{lr:g}", lr=lr, steps=sweep_steps, seed=1,
                        lr_schedule="cosine", no_monitor=True, **cfg, **extra)
            lr = _pick_best(f"sweep_{reg}_{arm}_lr", grid)
            for seed in [1, 2, 3, 4, 5, 6]:
                run_one(f"final_{reg}_{arm}_s{seed}", lr=lr, steps=final_steps, seed=seed,
                        lr_schedule="cosine", no_monitor=True, **cfg, **extra)


def stage_lm_ext():
    """Grid-edge extension for Phase 2 (protocol rule: never conclude from an edge lr).
    Widens each edge-picking arm's sweep by one step on the edge side; any arm whose
    widened-grid best lr moves gets fresh 6-seed finals under regime FAx/FMx."""
    cells = {
        "FA": ({"arch": "llama", "dataset": "fineweb", "batch": 16, "block": 1024,
                "n_layer": 12, "n_head": 12, "dim": 768}, 1250, 3000),
        "FM": ({"arch": "mamba", "dataset": "fineweb", "batch": 8, "block": 1024,
                "n_layer": 20, "dim": 512}, 2500, 6000),
    }
    arm_extra = {
        "muoncos": {"optimizer": "muon"},
        "auto2": {"optimizer": "tcg", "gate_mode": "normal", "auto_gate": True,
                  "auto_version": 2},
        "adamwcos": {"optimizer": "adamw"},
    }
    ext = [
        ("FA", "muoncos", [0.04], [5e-3, 1e-2, 2e-2, 0.04]),
        ("FA", "auto2", [0.04], [5e-3, 1e-2, 2e-2, 0.04]),
        ("FM", "muoncos", [2.5e-3], [2.5e-3, 5e-3, 1e-2, 2e-2]),
        ("FM", "auto2", [2.5e-3], [2.5e-3, 5e-3, 1e-2, 2e-2]),
        ("FM", "adamwcos", [2.4e-3], [3e-4, 6e-4, 1.2e-3, 2.4e-3]),
    ]
    for reg, arm, new_lrs, grid in ext:
        cfg, sweep_steps, final_steps = cells[reg]
        for lr in new_lrs:
            run_one(f"sweep_{reg}_{arm}_lr{lr:g}", lr=lr, steps=sweep_steps, seed=1,
                    lr_schedule="cosine", no_monitor=True, **cfg, **arm_extra[arm])
        best = _pick_best(f"sweep_{reg}_{arm}_lr", grid)
        if best in new_lrs:
            print(f"EXT: {reg}/{arm} best lr moved to {best} -> fresh finals as {reg}x")
            for seed in [1, 2, 3, 4, 5, 6]:
                run_one(f"final_{reg}x_{arm}_s{seed}", lr=best, steps=final_steps,
                        seed=seed, lr_schedule="cosine", no_monitor=True,
                        **cfg, **arm_extra[arm])
        else:
            print(f"EXT: {reg}/{arm} best lr stays {best} — edge was a false alarm")


def stage_fast_profile():
    """Overhead tuning for EchoMuon: 'fast' profile — gate basis refresh 25->100 steps
    (4x fewer amortized eigh) and memorization probe 100->200 (half the extra fwds) —
    at the flagship cells FA (LLaMA/FineWeb LM) and TIA (Tiny-ImageNet vision), same
    best lrs, 6/8 paired seeds vs the existing standard-profile finals.
    Pre-registered prediction: quality within noise of standard auto2 (the consistency
    basis drifts slowly; lambda sits at ~1.0 in both cells so coarser probes are
    inert), wall overhead FA +21% -> ~8-12%, TIA +46% -> ~15-25% (the per-step cached
    projector matmuls are not touched by these knobs)."""
    fast = {"optimizer": "tcg", "gate_mode": "normal", "auto_gate": True,
            "auto_version": 2, "gate_every": 100, "probe_every": 200}
    lr_fa = _pick_best("sweep_FA_auto2_lr", [5e-3, 1e-2, 2e-2, 0.04])
    for seed in [1, 2, 3, 4, 5, 6]:
        run_one(f"final_FA_auto2f_s{seed}", lr=lr_fa, steps=3000, seed=seed,
                arch="llama", dataset="fineweb", batch=16, block=1024,
                n_layer=12, n_head=12, dim=768,
                lr_schedule="cosine", no_monitor=True, **fast)
    lr_ti = _pick_best("sweep_TIA_auto2_lr", V_GRID)
    for seed in CTL_SEEDS:
        run_one(f"final_TIA_auto2f_s{seed}", lr=lr_ti, task="vision",
                dataset="tinyimagenet", batch=128, steps=V_FINAL_STEPS, seed=seed,
                lr_schedule="cosine", no_monitor=True, **V_DIMS, **fast)


def stage_fast_noise():
    """The last fast-profile gap: label-noise regime (TIP), where the gate does its
    most active damping — does the 4x-staler basis keep the +1.53% noise win?
    Same best lr as the standard-profile TIP auto2 finals, 8 paired seeds."""
    fast = {"optimizer": "tcg", "gate_mode": "normal", "auto_gate": True,
            "auto_version": 2, "gate_every": 100, "probe_every": 200}
    lr = _pick_best("sweep_TIP_auto2_lr", V_GRID)
    for seed in CTL_SEEDS:
        run_one(f"final_TIP_auto2f_s{seed}", lr=lr, task="vision",
                dataset="tinyimagenetn20", batch=128, steps=V_FINAL_STEPS, seed=seed,
                lr_schedule="cosine", no_monitor=True, **V_DIMS, **fast)


def stage_vision_strong():
    """Strong-recipe spot check: same compact ViT (identical-Block design preserved),
    but RandAugment(2,9) + mixup(0.2) + label smoothing(0.1) and 4x the budget
    (24000 steps ~ 61 CIFAR epochs). EchoMuon runs its fast profile (the recommended
    default). Question: does the +1-2pp margin over scheduled Muon survive at
    92-95% / ~70% accuracy territory?
    Pre-registered predictions: (a) mixup+RA absorb part of the gate's niche, so the
    margin SHRINKS; (b) it stays positive and significant on S100 (more headroom),
    S10 may compress toward a tie near its ceiling; (c) mixup dilutes the
    memorization signal, so lambda drops below the 0.95-1.00 seen so far, pulling
    EchoMuon toward plain Muon — the structural never-worse protection."""
    arms = [
        ("muoncos", {"optimizer": "muon"}, V_GRID),
        ("echomuonf", {"optimizer": "tcg", "gate_mode": "normal", "auto_gate": True,
                       "auto_version": 2, "gate_every": 100, "probe_every": 200}, V_GRID),
        ("adamwcos", {"optimizer": "adamw"}, [3e-4, 1e-3, 3e-3]),
    ]
    recipe = {"randaugment": True, "mixup": 0.2, "label_smoothing": 0.1}
    for reg, ds in [("S10", "cifar10"), ("S100", "cifar100")]:
        for arm, extra, grid in arms:
            for lr in grid:
                run_one(f"sweep_{reg}_{arm}_lr{lr:g}", lr=lr, task="vision", dataset=ds,
                        batch=128, steps=6000, seed=1, lr_schedule="cosine",
                        no_monitor=True, **V_DIMS, **recipe, **extra)
            lr = _pick_best(f"sweep_{reg}_{arm}_lr", grid)
            for seed in [1, 2, 3, 4]:
                run_one(f"final_{reg}_{arm}_s{seed}", lr=lr, task="vision", dataset=ds,
                        batch=128, steps=24000, seed=seed, lr_schedule="cosine",
                        no_monitor=True, **V_DIMS, **recipe, **extra)


def stage_vision_strong_ext():
    """Grid-edge extension for the strong-recipe cells: every arm picked the LOW edge
    (stronger regularization + 4x budget shifts optima down). Widen one step down;
    any arm whose widened-grid best moves gets fresh 4-seed finals as S10x/S100x."""
    arm_extra = {
        "muoncos": {"optimizer": "muon"},
        "echomuonf": {"optimizer": "tcg", "gate_mode": "normal", "auto_gate": True,
                      "auto_version": 2, "gate_every": 100, "probe_every": 200},
        "adamwcos": {"optimizer": "adamw"},
    }
    grids = {"muoncos": ([5e-3], [5e-3, 0.01, 0.02, 0.05]),
             "echomuonf": ([5e-3], [5e-3, 0.01, 0.02, 0.05]),
             "adamwcos": ([1e-4], [1e-4, 3e-4, 1e-3, 3e-3])}
    recipe = {"randaugment": True, "mixup": 0.2, "label_smoothing": 0.1}
    for reg, ds in [("S10", "cifar10"), ("S100", "cifar100")]:
        for arm, (new_lrs, grid) in grids.items():
            for lr in new_lrs:
                run_one(f"sweep_{reg}_{arm}_lr{lr:g}", lr=lr, task="vision", dataset=ds,
                        batch=128, steps=6000, seed=1, lr_schedule="cosine",
                        no_monitor=True, **V_DIMS, **recipe, **arm_extra[arm])
            best = _pick_best(f"sweep_{reg}_{arm}_lr", grid)
            if best in new_lrs:
                print(f"EXT: {reg}/{arm} best lr moved to {best} -> fresh finals as {reg}x")
                for seed in [1, 2, 3, 4]:
                    run_one(f"final_{reg}x_{arm}_s{seed}", lr=best, task="vision",
                            dataset=ds, batch=128, steps=24000, seed=seed,
                            lr_schedule="cosine", no_monitor=True,
                            **V_DIMS, **recipe, **arm_extra[arm])
            else:
                print(f"EXT: {reg}/{arm} best lr stays {best} — edge was a false alarm")


def stage_vision_strong_ext2():
    """Recursive low-edge resolution for the strong-recipe cells. Per arm: extend the
    sweep downward (halving, max two more steps) until the optimum is interior, then
    run finals ONCE at the settled lr for any arm whose pick moved past the lr the
    S10x/S100x finals used (new finals land as S10y/S100y). Finally, top up the two
    Muon-family arms on the S100 regime to 8 seeds at their settled lr — the +0.95pp
    at n=4 (t=+1.50) needs power, and S100 is the strong-recipe headline cell."""
    arm_extra = {
        "muoncos": {"optimizer": "muon"},
        "echomuonf": {"optimizer": "tcg", "gate_mode": "normal", "auto_gate": True,
                      "auto_version": 2, "gate_every": 100, "probe_every": 200},
        "adamwcos": {"optimizer": "adamw"},
    }
    grids = {"muoncos": [5e-3, 0.01, 0.02, 0.05],
             "echomuonf": [5e-3, 0.01, 0.02, 0.05],
             "adamwcos": [1e-4, 3e-4, 1e-3, 3e-3]}
    x_lr = {"muoncos": 5e-3, "echomuonf": 5e-3, "adamwcos": 1e-4}
    recipe = {"randaugment": True, "mixup": 0.2, "label_smoothing": 0.1}
    settled = {}
    for reg, ds in [("S10", "cifar10"), ("S100", "cifar100")]:
        for arm, extra in arm_extra.items():
            grid = sorted(grids[arm])
            for _ in range(2):
                best = _pick_best(f"sweep_{reg}_{arm}_lr", grid)
                if best != grid[0]:
                    break
                lo = grid[0] / 2
                run_one(f"sweep_{reg}_{arm}_lr{lo:g}", lr=lo, task="vision", dataset=ds,
                        batch=128, steps=6000, seed=1, lr_schedule="cosine",
                        no_monitor=True, **V_DIMS, **recipe, **extra)
                grid = [lo] + grid
            best = _pick_best(f"sweep_{reg}_{arm}_lr", grid)
            settled[(reg, arm)] = best
            if best == grid[0]:
                print(f"EXT2 WARNING: {reg}/{arm} still at low edge after 2 halvings ({best})")
            if best != x_lr[arm]:
                print(f"EXT2: {reg}/{arm} settled lr {best} != finals lr -> fresh {reg}y finals")
                for seed in [1, 2, 3, 4]:
                    run_one(f"final_{reg}y_{arm}_s{seed}", lr=best, task="vision",
                            dataset=ds, batch=128, steps=24000, seed=seed,
                            lr_schedule="cosine", no_monitor=True,
                            **V_DIMS, **recipe, **extra)
            else:
                print(f"EXT2: {reg}/{arm} settled lr {best} == finals lr — S10x/S100x stand")
    # power top-up: S100 Muon-family arms to 8 seeds at their settled lr
    for arm in ["muoncos", "echomuonf"]:
        best = settled[("S100", arm)]
        tag = "S100x" if best == x_lr[arm] else "S100y"
        for seed in [5, 6, 7, 8]:
            run_one(f"final_{tag}_{arm}_s{seed}", lr=best, task="vision",
                    dataset="cifar100", batch=128, steps=24000, seed=seed,
                    lr_schedule="cosine", no_monitor=True,
                    **V_DIMS, **recipe, **arm_extra[arm])


def stage_gate_ablation():
    """Prior-art gate-statistic ablations on VP (CIFAR-10 + 20% label noise), the
    cheapest noisy cell. Four arms, each the full EchoMuon configuration (standard
    profile, memorization-gap controller where applicable) with ONLY the gate
    statistic swapped for a published alternative:
      ablmag - singular-MAGNITUDE gate (Soft Muon, jiakai.xyz 2026 / Pion,
               arXiv:2605.19282): identical median/floor/projector scaffolding.
      ablcos - per-matrix EMA-smoothed cossim(momentum, current grad) scalar
               (Magma, arXiv:2602.15322).
      ablcau - sign-agreement mask on the orthogonalized update, mean-normalized
               (Cautious, arXiv:2411.16085).
      ablmix - orthogonalize a fast+slow buffer MIXTURE, no gate (AdEMAMix,
               arXiv:2409.03137; the controller is inert - mixture always on).
    Same lr as the EchoMuon VP finals (sweep_VP_auto2_lr pick), same 8 paired seeds;
    pair directly against final_VP_auto2_s* and final_VP_muoncos_s*.
    Pre-registered predictions: (a) EchoMuon > ablmag - magnitude cannot separate a
    large noisy direction from a large clean one, which is exactly what 20% label
    noise manufactures; (b) ablcos and ablcau land between Muon and EchoMuon -
    agreement against the instantaneous gradient is the right sign but a noisier
    signal; (c) ablmix ~ Muon (tie) - blending adds memory but prices no direction."""
    lr = _pick_best("sweep_VP_auto2_lr", V_GRID)
    vp = {"task": "vision", "dataset": "cifar10n20", "batch": 128,
          "steps": V_FINAL_STEPS, **V_DIMS}
    au = {"auto_gate": True, "auto_version": 2}
    arms = [
        ("ablmag", {"gate_mode": "magnitude", **au}),
        ("ablcos", {"gate_mode": "cosgate", **au}),
        ("ablcau", {"gate_mode": "cautious", **au}),
        ("ablmix", {"gate_mode": "mix"}),
    ]
    for tag, extra in arms:
        for seed in CTL_SEEDS:
            run_one(f"final_VP_{tag}_s{seed}", optimizer="tcg", lr=lr, seed=seed,
                    lr_schedule="cosine", no_monitor=True, **vp, **extra)


def stage_gate_ablation_lm():
    """Dissociation test for the VP finding that magnitude ~ agreement under symmetric
    label noise: rerun the ablmag arm (magnitude driver in EchoMuon's scaffolding) on
    FA — LLaMA-162M / FineWeb-Edu, the natural-web-noise cell where EchoMuon's LM win
    lives (-31 mnats vs muoncos). Same lr as final_FA_auto2 (0.02 pick), standard
    profile, 6 paired seeds against the existing final_FA_{auto2,muoncos} runs.
    Pre-registered: on temporally structured natural noise the statistics DISSOCIATE —
    echomuon beats ablmag on final val loss (directions here can be large-but-
    inconsistent, which magnitude cannot see). If they tie again, the honest reading
    is that the agreement advantage is a non-significant trend everywhere measured
    and the scaffolding+controller carry the method; §8 will say whichever holds."""
    lr = _pick_best("sweep_FA_auto2_lr", [5e-3, 1e-2, 2e-2])
    fa = {"arch": "llama", "dataset": "fineweb", "batch": 16, "block": 1024,
          "n_layer": 12, "n_head": 12, "dim": 768}
    for seed in [1, 2, 3, 4, 5, 6]:
        run_one(f"final_FA_ablmag_s{seed}", optimizer="tcg", lr=lr, steps=3000,
                seed=seed, lr_schedule="cosine", no_monitor=True,
                gate_mode="magnitude", auto_gate=True, auto_version=2, **fa)


def stage_auto_pilot():
    """EchoMuon pilot: measured-overfitting gate interpolation. Four cells, schedule
    parity, own sweeps. Predictions: VP/VA lock toward lambda~1 and keep the vision
    wins; P and PscaleM stay at lambda~0 and tie muon exactly (never-worse)."""
    au = {"gate_mode": "normal", "auto_gate": True}
    cells = [
        ("PscaleM", {"dataset": "enwik8p10", "batch": 64, "steps": 4000,
                     "n_layer": 12, "n_head": 8, "dim": 512},
         [5e-3, 1e-2, 2e-2], 2000),
        ("P", {"dataset": "enwik8p10", "batch": 64, "steps": 4000}, [0.01, 0.02, 0.04], 2000),
        ("VP", {"task": "vision", "dataset": "cifar10n20", "batch": 128,
                "steps": V_FINAL_STEPS, **V_DIMS}, V_GRID, V_SWEEP_STEPS),
        ("VA", {"task": "vision", "dataset": "cifar10", "batch": 128,
                "steps": V_FINAL_STEPS, **V_DIMS}, V_GRID, V_SWEEP_STEPS),
    ]
    for cell, cfg, grid, sweep_steps in cells:
        steps = cfg.pop("steps")
        for lr in grid:
            run_one(f"sweep_{cell}_auto_lr{lr:g}", optimizer="tcg", lr=lr,
                    steps=sweep_steps, seed=1, lr_schedule="cosine",
                    no_monitor=True, **cfg, **au)
        lr = _pick_best(f"sweep_{cell}_auto_lr", grid)
        for seed in CTL_SEEDS:
            run_one(f"final_{cell}_auto_s{seed}", optimizer="tcg", lr=lr, steps=steps,
                    seed=seed, lr_schedule="cosine", no_monitor=True, **cfg, **au)


def stage_sm2():
    # B re-test: trend gating should be a no-op under stationary noise
    for lr in [0.01, 0.02, 0.04]:
        run_one(f"sm_sweep_B_sm2_lr{lr:g}", optimizer="muon", lr=lr, dataset=DATASET,
                steps=SWEEP_STEPS, grad_noise=REGIMES["B"], seed=1,
                spectral_lr=True, ctl_signals=SM2_SIGNALS)
    lr = _pick_best("sm_sweep_B_sm2_lr", [0.01, 0.02, 0.04])
    for seed in CTL_SEEDS:
        run_one(f"final_B_sm2_s{seed}", optimizer="muon", lr=lr, dataset=DATASET,
                steps=FINAL_STEPS, grad_noise=REGIMES["B"], seed=seed,
                spectral_lr=True, ctl_signals=SM2_SIGNALS)
    # D re-test: the win must survive the trend gating
    for lr in [2e-3, 5e-3, 1e-2]:
        run_one(f"ft_sweep_F2_sm2_lr{lr:g}", lr=lr, steps=FT_SWEEP_STEPS, seed=1,
                spectral_lr=True, ctl_signals=SM2_SIGNALS, **ft_kwargs("muon", "F2"))
    lr = _pick_best("ft_sweep_F2_sm2_lr", [2e-3, 5e-3, 1e-2])
    for seed in [1, 2, 3, 4, 5, 6]:
        run_one(f"ft_final_F2_sm2_s{seed}", lr=lr, steps=FT_FINAL_STEPS, seed=seed,
                spectral_lr=True, ctl_signals=SM2_SIGNALS, **ft_kwargs("muon", "F2"))
    # C2 fault injection: clean data, best clean lr, 20x lr fault at step 2000 for
    # 50 steps, grad clipping off so the fault actually bites
    lr = best_lr("A", "muon")
    for seed in CTL_SEEDS:
        run_one(f"final_C2_muon_s{seed}", optimizer="muon", lr=lr, dataset=DATASET,
                steps=FINAL_STEPS, grad_noise=0.0, grad_clip=0.0,
                lr_spike="2000:50:20", seed=seed)
        run_one(f"final_C2_sm2_s{seed}", optimizer="muon", lr=lr, dataset=DATASET,
                steps=FINAL_STEPS, grad_noise=0.0, grad_clip=0.0,
                lr_spike="2000:50:20", seed=seed, spectral_lr=True, ctl_signals=SM2_SIGNALS)
        run_one(f"final_C2_sm2novalve_s{seed}", optimizer="muon", lr=lr, dataset=DATASET,
                steps=FINAL_STEPS, grad_noise=0.0, grad_clip=0.0,
                lr_spike="2000:50:20", seed=seed, spectral_lr=True, ctl_signals="rank,conf2")


# ---------------- Natural-noise validation: does the noise win survive real noise? ----------------
# The entire "better under noise" column so far rests on injected Gaussian grad noise.
# N: small-batch pretraining (batch 8, NO injection) — the noise is genuine minibatch noise.
# P: poisoned data (enwik8 with 10% random train bytes, clean val) — data-space noise,
#    the realistic dirty-web-corpus scenario. Arms: muon, muon+ctl (rank), spectral-ctl2.
NOISE_REGIMES = {
    "N": {"batch": 8, "steps": 8000, "sweep_steps": 3000, "dataset": "enwik8",
          "grid": [5e-3, 1e-2, 2e-2]},
    "P": {"batch": 64, "steps": 4000, "sweep_steps": 2000, "dataset": "enwik8p10",
          "grid": [0.01, 0.02, 0.04]},
}


def stage_noise():
    for reg, cfg in NOISE_REGIMES.items():
        common = {"dataset": cfg["dataset"], "batch": cfg["batch"]}
        for lr in cfg["grid"]:
            run_one(f"sweep_{reg}_muon_lr{lr:g}", optimizer="muon", lr=lr,
                    steps=cfg["sweep_steps"], seed=1, no_monitor=True, **common)
        lr_m = _pick_best(f"sweep_{reg}_muon_lr", cfg["grid"])
        for lr in cfg["grid"]:
            run_one(f"sweep_{reg}_sm2_lr{lr:g}", optimizer="muon", lr=lr,
                    steps=cfg["sweep_steps"], seed=1, spectral_lr=True,
                    ctl_signals=SM2_SIGNALS, **common)
        lr_s = _pick_best(f"sweep_{reg}_sm2_lr", cfg["grid"])
        for seed in CTL_SEEDS:
            run_one(f"final_{reg}_muon_s{seed}", optimizer="muon", lr=lr_m,
                    steps=cfg["steps"], seed=seed, **common)
            run_one(f"final_{reg}_muonctl_s{seed}", optimizer="muon", lr=lr_m,
                    steps=cfg["steps"], seed=seed, spectral_lr=True, **common)
            run_one(f"final_{reg}_sm2_s{seed}", optimizer="muon", lr=lr_s,
                    steps=cfg["steps"], seed=seed, spectral_lr=True,
                    ctl_signals=SM2_SIGNALS, **common)


# ---------------- Vision generalization: ViT on CIFAR-10, clean + label noise ----------------
# Second modality/task with IDENTICAL optimizer code (ViT reuses the same Block, so the
# Muon-managed parameter set has the same structure). VA = clean CIFAR-10; VP = 20%
# symmetric label noise on train (clean test) — the standard real-noise vision benchmark.
VISION_REGIMES = {"VA": "cifar10", "VP": "cifar10n20"}
V_GRID = [0.01, 0.02, 0.05]
V_DIMS = {"n_layer": 6, "n_head": 4, "dim": 256}
V_SWEEP_STEPS = 2000
V_FINAL_STEPS = 6000


def stage_vision():
    for reg, ds in VISION_REGIMES.items():
        common = {"task": "vision", "dataset": ds, "batch": 128, **V_DIMS}
        for lr in V_GRID:
            run_one(f"sweep_{reg}_muon_lr{lr:g}", optimizer="muon", lr=lr,
                    steps=V_SWEEP_STEPS, seed=1, no_monitor=True, **common)
        lr_m = _pick_best(f"sweep_{reg}_muon_lr", V_GRID)
        for lr in V_GRID:
            run_one(f"sweep_{reg}_sm2_lr{lr:g}", optimizer="muon", lr=lr,
                    steps=V_SWEEP_STEPS, seed=1, spectral_lr=True,
                    ctl_signals=SM2_SIGNALS, **common)
        lr_s = _pick_best(f"sweep_{reg}_sm2_lr", V_GRID)
        for seed in CTL_SEEDS:
            run_one(f"final_{reg}_muon_s{seed}", optimizer="muon", lr=lr_m,
                    steps=V_FINAL_STEPS, seed=seed, **common)
            run_one(f"final_{reg}_muonctl_s{seed}", optimizer="muon", lr=lr_m,
                    steps=V_FINAL_STEPS, seed=seed, spectral_lr=True, **common)
            run_one(f"final_{reg}_sm2_s{seed}", optimizer="muon", lr=lr_s,
                    steps=V_FINAL_STEPS, seed=seed, spectral_lr=True,
                    ctl_signals=SM2_SIGNALS, **common)


# ---------------- fine-tuning protocol (Idea 1 repositioned) ----------------
# Base: Muon-pretrained on enwik8 (~1 epoch). Fine-tune on 1M-token byte-level
# Shakespeare for 3000 steps (F1: ~50 epochs) — the natural memorization regime where
# the SNR-gate hypothesis predicts ShrunkMuon needs no early stopping and forgets less.
# Fair competitors are regularizers, hence the muon+wd arm.
FT_DATASET = "shakespeare_bytes"
BASE_CKPT = os.path.join(RESULTS, "base", "base_muon_enwik8.pt")
FT_ARMS = {
    "muon": {"optimizer": "muon", "grid": [2e-3, 5e-3, 1e-2]},
    "muonwd": {"optimizer": "muon", "grid": [2e-3, 5e-3, 1e-2], "muon_wd": 0.1},
    "shrunk": {"optimizer": "shrunk", "grid": [5e-3, 1e-2, 2e-2]},
    "adamw": {"optimizer": "adamw", "grid": [1e-4, 3e-4, 1e-3]},
}
FT_REGIMES = {"F1": 64, "F2": 8}  # name -> batch size
FT_SWEEP_STEPS = 1500
FT_FINAL_STEPS = 3000


def stage_pretrain():
    run_one("pretrain_base", optimizer="muon", lr=0.01, dataset="enwik8", steps=6000,
            seed=1, no_monitor=True, save_checkpoint=BASE_CKPT)


def ft_kwargs(arm: str, regime: str):
    cfg = FT_ARMS[arm]
    kw = {"optimizer": cfg["optimizer"], "dataset": FT_DATASET, "extra_val": "enwik8",
          "init_from": BASE_CKPT, "batch": FT_REGIMES[regime]}
    if "muon_wd" in cfg:
        kw["muon_wd"] = cfg["muon_wd"]
    return kw


def stage_ft_sweep():
    for regime in FT_REGIMES:
        for arm, cfg in FT_ARMS.items():
            for lr in cfg["grid"]:
                run_one(f"ft_sweep_{regime}_{arm}_lr{lr:g}", lr=lr, steps=FT_SWEEP_STEPS,
                        seed=1, no_monitor=True, **ft_kwargs(arm, regime))


def ft_best_lr(regime: str, arm: str):
    runs = []
    for lr in FT_ARMS[arm]["grid"]:
        p = os.path.join(RESULTS, "runs", f"ft_sweep_{regime}_{arm}_lr{lr:g}", "final.json")
        if os.path.exists(p):
            with open(p) as f:
                runs.append((json.load(f)["final_val"], lr))
    if not runs:
        raise RuntimeError(f"no ft-sweep results for {regime}/{arm}; run ft-sweep first")
    runs.sort()
    val, lr = runs[0]
    if lr in (FT_ARMS[arm]["grid"][0], FT_ARMS[arm]["grid"][-1]):
        print(f"WARNING: best ft lr for {regime}/{arm} is at grid edge ({lr})")
    return lr


def stage_ft_final():
    for regime in FT_REGIMES:
        for arm in FT_ARMS:
            lr = ft_best_lr(regime, arm)
            for seed in FINAL_SEEDS:
                run_one(f"ft_final_{regime}_{arm}_s{seed}", lr=lr, steps=FT_FINAL_STEPS,
                        seed=seed, **ft_kwargs(arm, regime))


def stage_analyze():
    r = subprocess.run([sys.executable, "-m", "src.analyze"])
    if r.returncode != 0:
        raise RuntimeError("analyze failed")


# ================= v2 revision campaign (the paper's named experimental debt) =================
# Each stage is independently resumable (final.json skip). Queue order puts the
# Table-2 blocker first, then the framing-deciding and control experiments, then
# the wider sweeps. GPU cost notes are per-stage docstrings.
AU2 = {"optimizer": "tcg", "gate_mode": "normal", "auto_gate": True, "auto_version": 2}
FA_CFG = {"arch": "llama", "dataset": "fineweb", "batch": 16, "block": 1024,
          "n_layer": 12, "n_head": 12, "dim": 768}
PS_DIMS = {"M": {"n_layer": 12, "n_head": 8, "dim": 512},
           "L": {"n_layer": 16, "n_head": 12, "dim": 768}}
PS_SEEDS = {"M": list(range(1, 17)), "L": list(range(1, 7))}


def _pick_best_glob(prefix):
    """Best lr over EVERY completed sweep run matching prefix (the settled grid)."""
    runs = []
    rd = os.path.join(RESULTS, "runs")
    for d in os.listdir(rd):
        if d.startswith(prefix):
            try:
                lr = float(d[len(prefix):])
            except ValueError:
                continue
            p = os.path.join(rd, d, "final.json")
            if os.path.exists(p):
                with open(p) as f:
                    runs.append((json.load(f)["final_val"], lr))
    if not runs:
        raise RuntimeError(f"no sweep results at {prefix}*")
    runs.sort()
    return runs[0][1]


def stage_v2_byte_grid():
    """Debt (1), BLOCKER for Table 2: the byte 38M/114M EchoMuon arms picked the TOP
    of [5e-3,1e-2,2e-2] while Muon picked interior 0.01 — the concession rows are
    edge-vs-interior. Widen upward (doubling, max twice) for any arm at the top edge;
    if the settled best leaves the old grid, re-run finals as PscaleMw/PscaleLw
    (n=16/6). ~40 min of sweeps; +3.7 GPU-h iff the EchoMuon pick moves."""
    grid0 = [5e-3, 1e-2, 2e-2]
    for size in ["M", "L"]:
        dims = PS_DIMS[size]
        for arm, extra in [("auto2", AU2), ("muoncos", {"optimizer": "muon"})]:
            grid = list(grid0)
            for _ in range(2):
                best = _pick_best(f"sweep_Pscale{size}_{arm}_lr", grid)
                if best != grid[-1]:
                    break
                hi = grid[-1] * 2
                run_one(f"sweep_Pscale{size}_{arm}_lr{hi:g}", lr=hi, dataset="enwik8p10",
                        batch=64, steps=2000, seed=1, lr_schedule="cosine",
                        no_monitor=True, **dims, **extra)
                grid = grid + [hi]
            best = _pick_best(f"sweep_Pscale{size}_{arm}_lr", grid)
            if best not in grid0:
                print(f"V2: Pscale{size}/{arm} best lr moved to {best} -> finals as Pscale{size}w")
                for seed in PS_SEEDS[size]:
                    run_one(f"final_Pscale{size}w_{arm}_s{seed}", lr=best,
                            dataset="enwik8p10", batch=64, steps=4000, seed=seed,
                            lr_schedule="cosine", no_monitor=True, **dims, **extra)
            else:
                print(f"V2: Pscale{size}/{arm} best lr stays {best} — edge resolved in place")


V2_VCELLS = {"VA": "cifar10", "VP": "cifar10n20", "TIA": "tinyimagenet",
             "TIP": "tinyimagenetn20"}


def stage_v2_horizon():
    """Debt (4): the true-horizon ring — append once per probe so re-seen batches are
    genuinely 400-700 steps old (the paper's original intent) instead of 4-7. Echo
    arm only, same lr as the auto2 finals; pairs against existing auto2 and muoncos.
    Decides whether v3 keeps the retention framing or upgrades it. ~4.5 GPU-h."""
    for reg, ds in V2_VCELLS.items():
        lr = _pick_best_glob(f"sweep_{reg}_auto2_lr")
        for seed in CTL_SEEDS:
            run_one(f"final_{reg}_auto2h_s{seed}", lr=lr, task="vision", dataset=ds,
                    batch=128, steps=V_FINAL_STEPS, seed=seed, lr_schedule="cosine",
                    no_monitor=True, ring_per_probe=True, **V_DIMS, **AU2)
    lr = _pick_best_glob("sweep_FA_auto2_lr")
    for seed in range(1, 7):
        run_one(f"final_FA_auto2h_s{seed}", lr=lr, steps=3000, seed=seed,
                lr_schedule="cosine", no_monitor=True, ring_per_probe=True,
                **FA_CFG, **AU2)


def stage_v2_controls():
    """Debt (3): norm-matched controls on the HEADLINE cells (they exist only on the
    two small always-on cells today). (a) shuffle with the full lambda controller on
    TIA/TIP/FA; (b) scalar-shrink (same realized per-layer Frobenius contraction,
    zero directional content) on VP/TIA/FA. ~5 GPU-h."""
    for reg, ds in [("TIA", "tinyimagenet"), ("TIP", "tinyimagenetn20")]:
        lr = _pick_best_glob(f"sweep_{reg}_auto2_lr")
        for seed in CTL_SEEDS:
            run_one(f"final_{reg}_shufauto_s{seed}", lr=lr, task="vision", dataset=ds,
                    batch=128, steps=V_FINAL_STEPS, seed=seed, lr_schedule="cosine",
                    no_monitor=True, optimizer="tcg", gate_mode="shuffled",
                    auto_gate=True, auto_version=2, **V_DIMS)
    lr = _pick_best_glob("sweep_FA_auto2_lr")
    for seed in range(1, 7):
        run_one(f"final_FA_shufauto_s{seed}", lr=lr, steps=3000, seed=seed,
                lr_schedule="cosine", no_monitor=True, optimizer="tcg",
                gate_mode="shuffled", auto_gate=True, auto_version=2, **FA_CFG)
    for reg, ds in [("VP", "cifar10n20"), ("TIA", "tinyimagenet")]:
        lr = _pick_best_glob(f"sweep_{reg}_auto2_lr")
        for seed in CTL_SEEDS:
            run_one(f"final_{reg}_sclshr_s{seed}", lr=lr, task="vision", dataset=ds,
                    batch=128, steps=V_FINAL_STEPS, seed=seed, lr_schedule="cosine",
                    no_monitor=True, optimizer="tcg", gate_mode="scalar",
                    auto_gate=True, auto_version=2, **V_DIMS)
    lr = _pick_best_glob("sweep_FA_auto2_lr")
    for seed in range(1, 7):
        run_one(f"final_FA_sclshr_s{seed}", lr=lr, steps=3000, seed=seed,
                lr_schedule="cosine", no_monitor=True, optimizer="tcg",
                gate_mode="scalar", auto_gate=True, auto_version=2, **FA_CFG)


def stage_v2_clean38():
    """Debt (2a): clean-vs-corrupted enwik8 at 38M — the missing contrast that turns
    'vanishes on byte text' into a measured statement about corruption vs tokenization.
    Own sweeps (widen on either edge, max twice), finals n=8 both arms. ~3 GPU-h."""
    grid0 = [5e-3, 1e-2, 2e-2]
    dims = PS_DIMS["M"]
    for arm, extra in [("muoncos", {"optimizer": "muon"}), ("auto2", AU2)]:
        grid = list(grid0)
        for lr in grid0:
            run_one(f"sweep_PcleanM_{arm}_lr{lr:g}", lr=lr, dataset="enwik8", batch=64,
                    steps=2000, seed=1, lr_schedule="cosine", no_monitor=True,
                    **dims, **extra)
        for _ in range(2):
            best = _pick_best(f"sweep_PcleanM_{arm}_lr", grid)
            if best == grid[-1]:
                new = grid[-1] * 2
                grid = grid + [new]
            elif best == grid[0]:
                new = grid[0] / 2
                grid = [new] + grid
            else:
                break
            run_one(f"sweep_PcleanM_{arm}_lr{new:g}", lr=new, dataset="enwik8",
                    batch=64, steps=2000, seed=1, lr_schedule="cosine",
                    no_monitor=True, **dims, **extra)
        best = _pick_best(f"sweep_PcleanM_{arm}_lr", grid)
        for seed in range(1, 9):
            run_one(f"final_PcleanM_{arm}_s{seed}", lr=best, dataset="enwik8",
                    batch=64, steps=4000, seed=seed, lr_schedule="cosine",
                    no_monitor=True, **dims, **extra)


def stage_v2_noise_dose():
    """Debt (5a): injected-noise dose response, 10%/40% on CIFAR-10 and CIFAR-100
    (0%/20% exist). Grid starts one step below the old bottom edge and widens down.
    ~4 GPU-h."""
    cells = [("V10d10", "cifar10n10"), ("V10d40", "cifar10n40"),
             ("V100d10", "cifar100n10"), ("V100d40", "cifar100n40")]
    grid0 = [5e-3, 0.01, 0.02, 0.05]
    for reg, ds in cells:
        common = {"task": "vision", "dataset": ds, "batch": 128, **V_DIMS}
        for arm, extra in [("muoncos", {"optimizer": "muon"}), ("auto2", AU2)]:
            grid = list(grid0)
            for lr in grid0:
                run_one(f"sweep_{reg}_{arm}_lr{lr:g}", lr=lr, steps=V_SWEEP_STEPS,
                        seed=1, lr_schedule="cosine", no_monitor=True, **common, **extra)
            for _ in range(2):
                best = _pick_best(f"sweep_{reg}_{arm}_lr", grid)
                if best != grid[0]:
                    break
                lo = grid[0] / 2
                run_one(f"sweep_{reg}_{arm}_lr{lo:g}", lr=lo, steps=V_SWEEP_STEPS,
                        seed=1, lr_schedule="cosine", no_monitor=True, **common, **extra)
                grid = [lo] + grid
            best = _pick_best(f"sweep_{reg}_{arm}_lr", grid)
            for seed in CTL_SEEDS:
                run_one(f"final_{reg}_{arm}_s{seed}", lr=best, steps=V_FINAL_STEPS,
                        seed=seed, lr_schedule="cosine", no_monitor=True,
                        **common, **extra)


def stage_v2_lambda_ladder():
    """Never-worse quantification on the losing byte-38M cell: pin lambda at
    0.25/0.5/0.75 (controller off) at the settled auto2 lr — how far is the
    controller's ~0.93 from the lambda that would have avoided the loss? n=8 each.
    ~3.5 GPU-h."""
    lr = _pick_best_glob("sweep_PscaleM_auto2_lr")
    for tag, lam in [("lam25", 0.25), ("lam50", 0.5), ("lam75", 0.75)]:
        for seed in CTL_SEEDS:
            run_one(f"final_PscaleM_{tag}_s{seed}", optimizer="tcg", gate_mode="normal",
                    fixed_lambda=lam, lr=lr, dataset="enwik8p10", batch=64, steps=4000,
                    seed=seed, lr_schedule="cosine", no_monitor=True, **PS_DIMS["M"])


def stage_v2_valsplit():
    """Debt (6): vision lr re-selection on a held-out 10% validation split, grid
    widened below the old bottom edge; selection never touches the test set. Cells
    whose val-selected pick differs from the shipped 0.01 get fresh test-reported
    finals as {cell}v. ~1.5 GPU-h of sweeps; finals only if picks move."""
    cells = {"VA": "cifar10", "VP": "cifar10n20", "V100A": "cifar100",
             "V100P": "cifar100n20", "TIA": "tinyimagenet", "TIP": "tinyimagenetn20"}
    grid0 = [2.5e-3, 5e-3, 0.01, 0.02, 0.05]
    for reg, ds in cells.items():
        common = {"task": "vision", "dataset": ds, "batch": 128, **V_DIMS}
        for arm, extra in [("muoncos", {"optimizer": "muon"}), ("auto2", AU2)]:
            grid = list(grid0)
            for lr in grid0:
                run_one(f"sweepv_{reg}_{arm}_lr{lr:g}", lr=lr, steps=V_SWEEP_STEPS,
                        seed=1, lr_schedule="cosine", no_monitor=True, val_frac=0.1,
                        **common, **extra)
            for _ in range(2):
                best = _pick_best(f"sweepv_{reg}_{arm}_lr", grid)
                if best != grid[0]:
                    break
                lo = grid[0] / 2
                run_one(f"sweepv_{reg}_{arm}_lr{lo:g}", lr=lo, steps=V_SWEEP_STEPS,
                        seed=1, lr_schedule="cosine", no_monitor=True, val_frac=0.1,
                        **common, **extra)
                grid = [lo] + grid
            best = _pick_best(f"sweepv_{reg}_{arm}_lr", grid)
            if best != 0.01:
                print(f"V2: {reg}/{arm} val-selected lr {best} != 0.01 -> finals as {reg}v")
                for seed in CTL_SEEDS:
                    run_one(f"final_{reg}v_{arm}_s{seed}", lr=best, steps=V_FINAL_STEPS,
                            seed=seed, lr_schedule="cosine", no_monitor=True,
                            **common, **extra)
            else:
                print(f"V2: {reg}/{arm} val-selected lr stays 0.01 — shipped finals stand")


def stage_v2_budget_match():
    """The missing strong-recipe control: LIGHT recipe at the strong budget (24000
    steps) with the FAST profile — separates recipe from budget/profile in the §5
    compression story. Own sweeps (6000 steps), finals n=8. ~7.5 GPU-h."""
    fastp = {**AU2, "gate_every": 100, "probe_every": 200}
    grid0 = [2.5e-3, 5e-3, 0.01, 0.02]
    for reg, ds in [("B10", "cifar10"), ("B100", "cifar100")]:
        common = {"task": "vision", "dataset": ds, "batch": 128, **V_DIMS}
        for arm, extra in [("muoncos", {"optimizer": "muon"}), ("echomuonf", fastp)]:
            grid = list(grid0)
            for lr in grid0:
                run_one(f"sweep_{reg}_{arm}_lr{lr:g}", lr=lr, steps=6000, seed=1,
                        lr_schedule="cosine", no_monitor=True, **common, **extra)
            for _ in range(2):
                best = _pick_best(f"sweep_{reg}_{arm}_lr", grid)
                if best != grid[0]:
                    break
                lo = grid[0] / 2
                run_one(f"sweep_{reg}_{arm}_lr{lo:g}", lr=lo, steps=6000, seed=1,
                        lr_schedule="cosine", no_monitor=True, **common, **extra)
                grid = [lo] + grid
            best = _pick_best(f"sweep_{reg}_{arm}_lr", grid)
            for seed in CTL_SEEDS:
                run_one(f"final_{reg}_{arm}_s{seed}", lr=best, steps=24000, seed=seed,
                        lr_schedule="cosine", no_monitor=True, **common, **extra)


def stage_v2_fa3x():
    """Debt (8): token-budget ladder on the LLaMA cell — 3x the paper's budget
    (9000 steps = 147M tokens, 0.91 tok/param), own sweeps at 3750 steps because the
    cosine is budget-normalized. n=3 paired. ~5 GPU-h."""
    grid = [1e-2, 2e-2, 4e-2]
    for arm, extra in [("muoncos", {"optimizer": "muon"}), ("auto2", AU2)]:
        for lr in grid:
            run_one(f"sweep_FA3x_{arm}_lr{lr:g}", lr=lr, steps=3750, seed=1,
                    lr_schedule="cosine", no_monitor=True, **FA_CFG, **extra)
        best = _pick_best(f"sweep_FA3x_{arm}_lr", grid)
        for seed in [1, 2, 3]:
            run_one(f"final_FA3x_{arm}_s{seed}", lr=best, steps=9000, seed=seed,
                    lr_schedule="cosine", no_monitor=True, **FA_CFG, **extra)


def stage_v2_resweep_cifar():
    """Protocol parity + robustness of the vision lr picks. The CIFAR sweepv runs
    selected on a CONTIGUOUS train tail; Tiny ImageNet needed a shuffled split (its
    on-disk order is class-sorted, so a contiguous tail was class-disjoint). Re-select
    the four CIFAR cells under the shuffled split so one protocol covers every cell,
    and test the one fragile pick (V100A/auto2 won by 0.0026 at n=1). New run-ids, so
    the original sweeps survive as evidence that the split method does not move the
    pick. Finals only for arm-cells whose pick actually moves. ~45 min + conditionals.
    """
    cells = {"VA": "cifar10", "VP": "cifar10n20",
             "V100A": "cifar100", "V100P": "cifar100n20"}
    grid = [2.5e-3, 5e-3, 0.01, 0.02, 0.05]
    for reg, ds in cells.items():
        common = {"task": "vision", "dataset": ds, "batch": 128, **V_DIMS}
        for arm, extra in [("muoncos", {"optimizer": "muon"}), ("auto2", AU2)]:
            for lr in grid:
                run_one(f"sweepw_{reg}_{arm}_lr{lr:g}", lr=lr, steps=V_SWEEP_STEPS,
                        seed=1, lr_schedule="cosine", no_monitor=True, val_frac=0.1,
                        **common, **extra)
            new = _pick_best(f"sweepw_{reg}_{arm}_lr", grid)
            old = _pick_best(f"sweepv_{reg}_{arm}_lr", grid)
            if new == old:
                print(f"V2: {reg}/{arm} shuffled-split pick {new:g} == contiguous pick "
                      f"{old:g} — selection is split-robust")
            else:
                print(f"V2: {reg}/{arm} shuffled-split pick {new:g} != contiguous pick "
                      f"{old:g} -> RE-RUNNING finals as {reg}w")
                for seed in CTL_SEEDS:
                    run_one(f"final_{reg}w_{arm}_s{seed}", lr=new, steps=V_FINAL_STEPS,
                            seed=seed, lr_schedule="cosine", no_monitor=True,
                            **common, **extra)


def _mean_val(run_ids):
    """Mean final_val over a set of completed runs (missing runs are ignored)."""
    vals = []
    for rid in run_ids:
        p = os.path.join(RESULTS, "runs", rid, "final.json")
        if os.path.exists(p):
            with open(p) as f:
                vals.append(json.load(f)["final_val"])
    return sum(vals) / len(vals) if vals else float("inf")


def stage_v2_fa3x_widen():
    """stage_v2_fa3x shipped the grid [0.01, 0.02, 0.04] with NO widening, and both
    arms picked its bottom edge -- the same defect this campaign exists to measure,
    and a shared too-high lr is known to flatter EchoMuon. Widen downward (max twice)
    and re-run the n=3 finals as FA3xw for any arm whose pick leaves 0.01.
    ~30 min if 0.01 survives; +3.7 GPU-h if it does not."""
    for arm, extra in [("muoncos", {"optimizer": "muon"}), ("auto2", AU2)]:
        grid = [1e-2, 2e-2, 4e-2]
        for _ in range(2):
            best = _pick_best(f"sweep_FA3x_{arm}_lr", grid)
            if best != grid[0]:
                break
            lo = grid[0] / 2
            run_one(f"sweep_FA3x_{arm}_lr{lo:g}", lr=lo, steps=3750, seed=1,
                    lr_schedule="cosine", no_monitor=True, **FA_CFG, **extra)
            grid = [lo] + grid
        best = _pick_best(f"sweep_FA3x_{arm}_lr", grid)
        if best == 1e-2:
            print(f"V2: FA3x/{arm} widened pick stays 0.01 -- shipped 3x finals stand")
        else:
            print(f"V2: FA3x/{arm} widened pick {best:g} != 0.01 -> finals as FA3xw")
            for seed in [1, 2, 3]:
                run_one(f"final_FA3xw_{arm}_s{seed}", lr=best, steps=9000, seed=seed,
                        lr_schedule="cosine", no_monitor=True, **FA_CFG, **extra)


V3_ARMS = [
    ("a1abs", {"gate_mode": "abscal", "auto_gate": True, "auto_version": 2,
               "probe_every": 100}),
    ("a2abshi", {"gate_mode": "abscal_hi", "auto_gate": True, "auto_version": 2,
                 "probe_every": 100}),
    ("a3sig", {"gate_mode": "normal", "auto_gate": True, "auto_version": 3}),
    ("a4both", {"gate_mode": "abscal", "auto_gate": True, "auto_version": 3}),
]
V3_SEEDS = [1, 2, 3]


def _v3_sweep_pick(prefix, grid, run, max_widen=2):
    """Sweep, then widen in whichever direction the pick lands on an edge.

    The paper's central methodological finding is that a grid edge is not a
    selection, so no arm here is allowed to be scored on one.
    """
    grid = sorted(grid)
    for lr in grid:
        run(lr)
    for _ in range(max_widen * 2):
        best = _pick_best(prefix, grid)
        if best == grid[0]:
            lo = grid[0] / 2
            run(lo)
            grid = [lo] + grid
        elif best == grid[-1]:
            hi = grid[-1] * 2
            run(hi)
            grid = grid + [hi]
        else:
            break
    best = _pick_best(prefix, grid)
    print(f"V3: {prefix} -> lr {best:g} "
          f"({'INTERIOR' if grid[0] < best < grid[-1] else 'STILL ON AN EDGE'})",
          flush=True)
    return best


def stage_v3_screen():
    """Screen the v3 gate arms on the two cells that define the hypothesis.

    Measured with lab/c_probe.py over 3,456 layer-observations: the consistency score
    c has a real absolute scale (its 10th percentile sits on the analytic pure-noise
    floor of 1.639 on all four cells measured), and the share of directions above the
    noise/signal midpoint separates the cells by outcome 5 to 1 -- 24.4% on Tiny
    ImageNet where EchoMuon wins against 4.6% on corrupted bytes where it loses. The
    median-relative rule throws that scale away, and because the median is computed
    over a set contaminated by the small-sigma ratio explosion (measured c up to 1775
    against a theoretical 5) it lands above the noise floor, so the shipped gate damps
    the highest-energy directions and passes the unstable ones.

    Arms:
      a1abs    absolute reference, renormalised to constant Frobenius norm
      a2abshi  the same, but the low-energy half is left ungated
      a3sig    shipped gate, lambda driven by the measured signal fraction
      a4both   both changes

    Pre-registered predictions, recorded before the runs:
      1. a1abs and a2abshi keep most of the Tiny ImageNet margin. If the absolute rule
         is right about which directions carry echo, damping the rest should not cost.
      2. a3sig turns the corrupted-byte loss into a tie or better, because the signal
         fraction there is 4.6% and lambda should fall accordingly.
      3. a4both is the best of the four on both cells, or the two changes interfere.
      4. a1abs and a2abshi select a learning rate near the baseline's, since the
         renormalisation is norm-preserving by construction. A large shift means the
         renormalisation is not doing what it claims.
    """
    cells = [
        ("TIA", {"task": "vision", "dataset": "tinyimagenet", "batch": 128, **V_DIMS},
         [2.5e-3, 5e-3, 0.01, 0.02, 0.05], V_SWEEP_STEPS, V_FINAL_STEPS,
         {"val_frac": 0.1}),
        ("PscaleM", {"task": "lm", "dataset": "enwik8p10", "batch": 64,
                     "n_layer": 12, "n_head": 8, "dim": 512, "block": 256},
         [5e-3, 0.01, 0.02, 0.04], 2000, 4000, {}),
    ]
    for cell, common, grid, sweep_steps, final_steps, sel in cells:
        for arm, extra in V3_ARMS:
            base = dict(optimizer="tcg", lr_schedule="cosine", no_monitor=True,
                        **common, **extra)
            prefix = f"sweep3_{cell}_{arm}_lr"

            def run(lr, _b=base, _p=prefix, _s=sweep_steps, _sel=sel):
                run_one(f"{_p}{lr:g}", lr=lr, steps=_s, seed=1, **_sel, **_b)

            best = _v3_sweep_pick(prefix, grid, run)
            for seed in V3_SEEDS:
                run_one(f"final3_{cell}_{arm}_s{seed}", lr=best, steps=final_steps,
                        seed=seed, **base)


def stage_v3_confirm():
    """Confirmatory n for the one arm the screen left standing: the gate-derived
    controller on the shipped gate (a3sig).

    The screen ran n=3, which cannot resolve the corrupted-byte cell: the published
    effect there is +0.0010 nats with sd 0.0019 over n=16, so |t|=2 needs about n=13.
    This fills a3sig out to the baselines' own seed counts, 8 on Tiny ImageNet and 16 on
    corrupted bytes, reusing the learning rates the screen already selected. abscal is
    not carried forward: it showed no gain on Tiny ImageNet and a measured harm on
    corrupted bytes of roughly four times the loss the shipped method already had.

    Two questions, both pre-registered:
      1. Does a3sig match the shipped method on Tiny ImageNet? The screen put it at
         -0.74 pp with t=-1.63, which is neither a match nor a difference at n=3.
      2. Does it remove the corrupted-byte loss? Unanswerable at n=3 and the reason
         this stage exists. lambda averaged 0.039 there against the shipped 0.927, so
         a3sig should behave close to plain Muon and the loss should go.
    """
    a3 = {"gate_mode": "normal", "auto_gate": True, "auto_version": 3}
    cells = [
        ("TIA", {"task": "vision", "dataset": "tinyimagenet", "batch": 128, **V_DIMS},
         [2.5e-3, 5e-3, 0.01, 0.02, 0.05], V_FINAL_STEPS, range(1, 9)),
        ("PscaleM", {"task": "lm", "dataset": "enwik8p10", "batch": 64,
                     "n_layer": 12, "n_head": 8, "dim": 512, "block": 256},
         [5e-3, 0.01, 0.02, 0.04], 4000, range(1, 17)),
    ]
    for cell, common, grid, steps, seeds in cells:
        lr = _pick_best(f"sweep3_{cell}_a3sig_lr", grid)
        print(f"V3-CONFIRM: {cell}/a3sig reusing lr {lr:g} from the screen", flush=True)
        for seed in seeds:
            rid = f"final3_{cell}_a3sig_s{seed}"
            if os.path.exists(os.path.join(RESULTS, "runs", rid, "final.json")):
                continue                       # the screen already ran seeds 1-3
            run_one(rid, optimizer="tcg", lr=lr, steps=steps, seed=seed,
                    lr_schedule="cosine", no_monitor=True, **common, **a3)


def stage_v2_multiseed_select():
    """Debt (9). Every lr grid in this project is scored from ONE seed, and the
    V100A/auto2 pick demonstrably flipped between two equally valid val splits with a
    1.45pp test consequence -- selection noise ~3x the CIFAR effect size. Re-select
    all four CIFAR cells from the MEAN val loss of 3 seeds; seed 1 is reused from the
    sweepw runs, so only seeds 2-3 are new (80 sweeps, not 120). Finals as {cell}m
    only where the 3-seed pick differs from the 1-seed pick. ~1.5 GPU-h + conditionals.
    The val split itself is seeded at 1234 independently of --seed, so this isolates
    training noise in the selection, holding the split fixed."""
    cells = {"VA": "cifar10", "VP": "cifar10n20",
             "V100A": "cifar100", "V100P": "cifar100n20"}
    grid = [2.5e-3, 5e-3, 0.01, 0.02, 0.05]
    seeds = [2, 3]
    for reg, ds in cells.items():
        common = {"task": "vision", "dataset": ds, "batch": 128, **V_DIMS}
        for arm, extra in [("muoncos", {"optimizer": "muon"}), ("auto2", AU2)]:
            for lr in grid:
                for seed in seeds:
                    run_one(f"sweepm_{reg}_{arm}_s{seed}_lr{lr:g}", lr=lr,
                            steps=V_SWEEP_STEPS, seed=seed, lr_schedule="cosine",
                            no_monitor=True, val_frac=0.1, **common, **extra)
            scored = []
            for lr in grid:
                ids = [f"sweepw_{reg}_{arm}_lr{lr:g}"] +                       [f"sweepm_{reg}_{arm}_s{s}_lr{lr:g}" for s in seeds]
                scored.append((_mean_val(ids), lr))
            scored.sort()
            best = scored[0][1]
            single = _pick_best(f"sweepw_{reg}_{arm}_lr", grid)
            gap = scored[1][0] - scored[0][0]
            if best == single:
                print(f"V2: {reg}/{arm} 3-seed pick {best:g} == 1-seed pick {single:g} "
                      f"(gap {gap:+.4f}) -- selection stable")
            else:
                print(f"V2: {reg}/{arm} 3-seed pick {best:g} != 1-seed pick {single:g} "
                      f"(gap {gap:+.4f}) -> finals as {reg}m")
                for seed in CTL_SEEDS:
                    run_one(f"final_{reg}m_{arm}_s{seed}", lr=best,
                            steps=V_FINAL_STEPS, seed=seed, lr_schedule="cosine",
                            no_monitor=True, **common, **extra)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    stages = {"smoke": [stage_smoke], "sweep": [stage_sweep], "final": [stage_final],
              "pretrain": [stage_pretrain], "ft-sweep": [stage_ft_sweep],
              "ft-final": [stage_ft_final], "ctl": [stage_ctl],
              "ctl-all": [stage_ctl, stage_analyze],
              "scale-sweep": [stage_scale_sweep], "scale-final": [stage_scale_final],
              "scale-all": [stage_scale_sweep, stage_scale_final, stage_analyze],
              "sm-sweep": [stage_sm_sweep], "sm-final": [stage_sm_final],
              "sm-all": [stage_sm_sweep, stage_sm_final, stage_analyze],
              "sm2-all": [stage_sm2, stage_analyze],
              "sm2-clean": [stage_sm2_clean, stage_analyze],
              "cos-control": [stage_cos_control, stage_analyze],
              "sched-parity": [stage_sched_parity, stage_analyze],
              "tcg-pilot": [stage_tcg_pilot, stage_analyze],
              "tcg-validate": [stage_tcg_validate, stage_analyze],
              "tcg-scale": [stage_tcg_scale, stage_analyze],
              "tcg-q20": [stage_tcg_q20, stage_analyze],
              "tcg-mfix": [stage_tcg_mfix, stage_analyze],
              "tcg-beta": [stage_tcg_beta, stage_analyze],
              "tcg-novelty": [stage_tcg_novelty, stage_analyze],
              "ccg-pilot": [stage_ccg_pilot, stage_analyze],
              "amp-pilot": [stage_amp_pilot, stage_analyze],
              "auto-pilot": [stage_auto_pilot, stage_analyze],
              "auto2-pilot": [stage_auto2_pilot, stage_analyze],
              "auto2-confirm": [stage_auto2_confirm, stage_analyze],
              "adamw-baselines": [stage_adamw_baselines, stage_analyze],
              "vision-breadth": [stage_vision_breadth, stage_analyze],
              "vision-tiny": [stage_vision_tiny, stage_analyze],
              "lm-breadth": [stage_lm_breadth, stage_analyze],
              "lm-ext": [stage_lm_ext],
              "fast-profile": [stage_fast_profile],
              "fast-noise": [stage_fast_noise],
              "vision-strong": [stage_vision_strong, stage_analyze],
              "vision-strong-ext": [stage_vision_strong_ext],
              "vision-strong-ext2": [stage_vision_strong_ext2],
              "gate-ablation": [stage_gate_ablation],
              "gate-ablation-lm": [stage_gate_ablation_lm],
              "v2-byte-grid": [stage_v2_byte_grid],
              "v2-horizon": [stage_v2_horizon],
              "v2-controls": [stage_v2_controls],
              "v2-clean38": [stage_v2_clean38],
              "v2-noise-dose": [stage_v2_noise_dose],
              "v2-lambda-ladder": [stage_v2_lambda_ladder],
              "v2-valsplit": [stage_v2_valsplit],
              "v2-budget-match": [stage_v2_budget_match],
              "v2-fa3x": [stage_v2_fa3x],
              "v2-resweep-cifar": [stage_v2_resweep_cifar],
              "v2-fa3x-widen": [stage_v2_fa3x_widen],
              "v2-multiseed": [stage_v2_multiseed_select],
              "v2-followup": [stage_v2_fa3x_widen, stage_v2_multiseed_select],
              "v3-screen": [stage_v3_screen],
              "v3-confirm": [stage_v3_confirm],
              "v2-queue": [stage_v2_byte_grid, stage_v2_horizon, stage_v2_controls,
                           stage_v2_clean38, stage_v2_noise_dose,
                           stage_v2_lambda_ladder, stage_v2_valsplit,
                           stage_v2_budget_match, stage_v2_fa3x,
                           stage_v2_resweep_cifar],
              "noise-all": [stage_noise, stage_analyze],
              "vision-all": [stage_vision, stage_analyze],
              "analyze": [stage_analyze],
              "all": [stage_smoke, stage_sweep, stage_final, stage_analyze],
              "ft-all": [stage_pretrain, stage_ft_sweep, stage_ft_final, stage_analyze]}
    if stage not in stages:
        print(f"unknown stage '{stage}'; choose from {list(stages)}")
        sys.exit(1)
    for fn in stages[stage]:
        fn()
