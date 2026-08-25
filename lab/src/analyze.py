"""Aggregate results -> results/report.md + results/plots/*.png.
Also runs the Idea-3 predictive analysis: does a jump in top singular value
precede a jump in training loss?"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = os.environ.get("RESULTS_DIR", "results")
PLOTS = os.path.join(RESULTS, "plots")


def load_finals():
    rows = []
    for p in glob.glob(os.path.join(RESULTS, "runs", "*", "final.json")):
        with open(p) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def load_log(run_id):
    rows = []
    p = os.path.join(RESULTS, "runs", run_id, "log.jsonl")
    if not os.path.exists(p):
        return pd.DataFrame()
    with open(p) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def regime_of(run_id):
    parts = run_id.split("_")
    allowed = ("A", "B", "C", "C2", "N", "P", "VA", "VP", "V100A", "V100P", "TIA", "TIP",
               "FA", "FM", "S10", "S100", "S10x", "S100x", "S10y", "S100y")
    return parts[1] if len(parts) > 1 and parts[1] in allowed else "?"


def arm_of(row):
    suffix = "+cos" if row.get("lr_schedule") == "cosine" else ""
    if row.get("spectral_lr"):
        sigs = row.get("ctl_signals")
        if isinstance(sigs, str) and sigs != "rank":
            # historical spectral-controller thread (working name "SpectralMuon",
            # relabeled spectral-ctl to avoid the SpecMuon arXiv collision)
            return {"rank,conf,valve": "spectral-ctl",
                    "rank,conf": "spectral-ctl-novalve",
                    "rank,conf2,valve": "spectral-ctl2",
                    "rank,conf2": "spectral-ctl2-novalve"}.get(
                        sigs, f"spectral-ctl[{sigs}]") + suffix
        mode = row.get("ctl_mode")
        if not isinstance(mode, str):  # runs predating the ctl_mode field used srank
            mode = "srank"
        return {"srank": "muon+ctl", "inverse": "muon+ctl-inv",
                "shuffled": "muon+ctl-shuf"}.get(mode, "muon+ctl") + suffix
    base = row["optimizer"]
    gm0 = row.get("gate_mode")
    if base == "tcg" and row.get("auto_gate") is True \
            and (not isinstance(gm0, str) or gm0 == "normal"):
        # gate-mode ablations run WITH the controller but are not EchoMuon —
        # fall through to the gate_mode labeling below instead
        av = row.get("auto_version")
        tag = "echomuon" if isinstance(av, (int, float)) and not pd.isna(av) and av == 2 \
            else "echomuon-v1"
        ge = row.get("gate_every")
        if (isinstance(ge, (int, float)) and not pd.isna(ge) and ge >= 100) \
                or "auto2f" in str(row.get("run_id", "")):
            tag += "-fast"  # fast profile (gate_every>=100); run-id fallback for
        return tag + suffix  # runs predating the gate_every field in final.json
    gm = row.get("gate_mode")
    if base == "tcg" and isinstance(gm, str) and gm != "normal":
        base += {"inverse": "-inv", "shuffled": "-shuf", "novelty": "-nov",
                 "coherence": "-ccg", "amplify": "-amp", "magnitude": "-mag",
                 "cosgate": "-cosg", "cautious": "-cau", "mix": "-mix"}.get(gm, f"-{gm}")
    if base == "tcg":
        if row.get("gate_stage") == "pre":
            base += "-pre"
        gb = row.get("gate_block")
        if isinstance(gb, (int, float)) and not pd.isna(gb) and gb > 1:
            base += f"-blk{int(gb)}"
        gq = row.get("gate_quantile")
        if isinstance(gq, float) and not pd.isna(gq) and gq != 0.5:
            base += f"-q{int(gq * 100)}"
    return base + suffix


def welch_table(dfinal, out):
    """Welch t-test of each arm vs plain muon on final_val, per regime.
    Positive t = the arm is better (lower loss) than muon."""
    recs = []
    for regime in sorted(dfinal.regime.unique()):
        d = dfinal[dfinal.regime == regime]
        base = d[d.arm == "muon"].final_val.dropna()
        if len(base) < 2:
            continue
        for arm in sorted(d.arm.unique()):
            if arm == "muon":
                continue
            x = d[d.arm == arm].final_val.dropna()
            if len(x) < 2:
                continue
            v1, v2 = base.var(ddof=1) / len(base), x.var(ddof=1) / len(x)
            t = (base.mean() - x.mean()) / np.sqrt(v1 + v2)
            df_w = (v1 + v2) ** 2 / (v1 ** 2 / (len(base) - 1) + v2 ** 2 / (len(x) - 1))
            recs.append({"regime": regime, "arm": arm, "delta_vs_muon": x.mean() - base.mean(),
                         "welch_t": t, "df": df_w, "n_arm": len(x), "n_muon": len(base)})
    if recs:
        t = pd.DataFrame(recs).set_index(["regime", "arm"])
        out.write("## Welch t-test vs plain muon (final_val)\n\n"
                  "Θετικό t = καλύτερο από muon. Χονδρικό όριο: |t|>2.2 ≈ p<0.05.\n\n"
                  + t.round(4).to_markdown() + "\n\n")


def sweep_table(df, out):
    d = df[df.run_id.str.startswith("sweep_")]
    if d.empty:
        return
    d = d.assign(regime=d.run_id.map(regime_of))
    t = d.pivot_table(index=["regime", "optimizer"], columns="lr", values="final_val")
    out.write("## Sweep: final val loss ανά lr\n\n" + t.to_markdown() + "\n\n")


def final_table(df, out):
    d = df[df.run_id.str.startswith("final_")]
    if d.empty:
        return None
    d = d.assign(regime=d.run_id.map(regime_of), arm=d.apply(arm_of, axis=1))
    aggs = dict(
        final_val_mean=("final_val", "mean"), final_val_std=("final_val", "std"),
        best_val_mean=("best_val", "mean"),
        diverged=("final_val", lambda x: int((x.isna() | (x > 5)).sum())),
        wall_s=("wall_s", "mean"),
        tok_per_s=("tok_per_s", "mean"), n=("seed", "count"), lr=("lr", "first"))
    if "final_acc" in d.columns and d.final_acc.notna().any():
        aggs["final_acc"] = ("final_acc", "mean")
        aggs["best_acc"] = ("best_acc", "mean")
    g = d.groupby(["regime", "arm"]).agg(**aggs)
    out.write("## Final: mean ± std across seeds (primary criterion: final_val_mean)\n\n"
              + g.round(4).to_markdown() + "\n\n")
    return d


def speed_table(dfinal, out):
    """Steps to reach 1.02x the best val loss achieved by any arm in the regime."""
    recs = []
    for regime in sorted(dfinal.regime.unique()):
        d = dfinal[dfinal.regime == regime]
        target = 1.02 * d.best_val.min()
        for _, row in d.iterrows():
            log = load_log(row.run_id)
            v = log.dropna(subset=["val_loss"]) if "val_loss" in log.columns else pd.DataFrame()
            hit = v[v.val_loss <= target].step.min() if not v.empty else np.nan
            recs.append({"regime": regime, "arm": row["arm"], "target": round(target, 4),
                         "steps_to_target": hit})
    t = pd.DataFrame(recs).groupby(["regime", "arm"]).agg(
        target=("target", "first"), steps_mean=("steps_to_target", "mean"),
        reached=("steps_to_target", lambda x: int(x.notna().sum())))
    out.write("## Speed: steps μέχρι 1.02× του best val (NaN = δεν το έφτασε)\n\n"
              + t.to_markdown() + "\n\n")


def curves_plot(dfinal):
    for regime in sorted(dfinal.regime.unique()):
        plt.figure(figsize=(8, 5))
        for arm in sorted(dfinal[dfinal.regime == regime].arm.unique()):
            runs = dfinal[(dfinal.regime == regime) & (dfinal.arm == arm)]
            curves = []
            for rid in runs.run_id:
                log = load_log(rid)
                v = log.dropna(subset=["val_loss"]) if "val_loss" in log else pd.DataFrame()
                if not v.empty:
                    curves.append(v.set_index("step")["val_loss"])
            if curves:
                m = pd.concat(curves, axis=1).mean(axis=1)
                plt.plot(m.index, m.values, label=arm, linewidth=2)
        plt.xlabel("step"); plt.ylabel("val loss"); plt.title(f"Regime {regime}")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, f"curves_{regime}.png"), dpi=130)
        plt.close()


def scale_section(df, out):
    """Idea-3 scale test: paired per-seed deltas (ctl - muon) across model sizes.
    Size S comes from the v4 final_B_{muon,muonctl} runs; M/L from scale_* runs."""
    d = df.copy()
    d["arm"] = d.apply(arm_of, axis=1)
    frames = []
    s = d[d.run_id.str.match(r"final_B_(muon|muonctl)_s\d+$")]
    if not s.empty:
        frames.append(s.assign(size="S-11M")[["size", "arm", "seed", "final_val"]])
    sc = d[d.run_id.str.startswith("scale_") & ~d.run_id.str.contains("_sweep_")]
    if not sc.empty:
        sc = sc.assign(size_key=sc.run_id.str.split("_").str[1])
        if "n_params" in sc.columns:
            label = {k: f"{k}-{g.n_params.dropna().iloc[0] / 1e6:.0f}M"
                     for k, g in sc.groupby("size_key") if g.n_params.notna().any()}
        else:
            label = {}
        sc = sc.assign(size=sc.size_key.map(lambda k: label.get(k, k)))
        frames.append(sc[["size", "arm", "seed", "final_val"]])
    if len(frames) < 2:
        return
    allf = pd.concat(frames)
    tbl = allf.groupby(["size", "arm"]).final_val.agg(["mean", "std", "count"])
    recs = []
    for size, g in allf.groupby("size"):
        piv = g.pivot_table(index="seed", columns="arm", values="final_val")
        if "muon" not in piv.columns or "muon+ctl" not in piv.columns:
            continue
        delta = (piv["muon+ctl"] - piv["muon"]).dropna()
        if len(delta) < 2:
            continue
        sem = delta.std(ddof=1) / np.sqrt(len(delta))
        recs.append({"size": size, "paired_delta": delta.mean(), "sem": sem,
                     "paired_t": delta.mean() / sem, "n_pairs": len(delta)})
    out.write("## Idea 3 scale test\n\n" + tbl.round(4).to_markdown() + "\n\n")
    if recs:
        def _size_key(s):
            try:
                return float(s.split("-")[-1].rstrip("M"))
            except ValueError:
                return float("inf")
        r = pd.DataFrame(recs).set_index("size")
        r = r.loc[sorted(r.index, key=_size_key)]
        out.write("Paired per-seed delta (ctl − muon)· αρνητικό = ctl καλύτερος. "
                  "df = n_pairs − 1.\n\n" + r.round(4).to_markdown() + "\n\n")
        plt.figure(figsize=(7, 4.5))
        plt.errorbar(range(len(r)), r["paired_delta"], yerr=r["sem"], fmt="o-", capsize=4)
        plt.axhline(0, color="gray", linewidth=1)
        plt.xticks(range(len(r)), r.index)
        plt.ylabel("Δ final val (ctl − muon)")
        plt.title("Controller effect vs model scale (regime B)")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, "scale_delta.png"), dpi=130)
        plt.close()
        out.write("Plot: `plots/scale_delta.png`\n\n")


def ft_section(df, out):
    """Fine-tuning protocol tables and curves. Run ids: ft_{sweep|final}_{regime}_{arm}_*"""
    d = df[df.run_id.str.startswith("ft_")].copy()
    if d.empty:
        return
    d["regime"] = d.run_id.str.split("_").str[2]
    # run-id segments are historical; only the displayed labels are renamed
    d["arm"] = d.run_id.str.split("_").str[3].replace(
        {"spectralmuon": "spectral-ctl", "sm2": "spectral-ctl2"})
    sw = d[d.run_id.str.startswith("ft_sweep_")]
    if not sw.empty:
        t = sw.pivot_table(index=["regime", "arm"], columns="lr", values="final_val")
        out.write("## FT sweep: final val (shakespeare) ανά lr\n\n" + t.round(4).to_markdown() + "\n\n")
    fi = d[d.run_id.str.startswith("ft_final_")]
    if fi.empty:
        return
    g = fi.groupby(["regime", "arm"]).agg(
        final_val=("final_val", "mean"), final_std=("final_val", "std"),
        best_val=("best_val", "mean"), retain_enwik8=("retain_val", "mean"),
        overfit_gap=("final_val", "mean"), wall_s=("wall_s", "mean"),
        n=("seed", "count"), lr=("lr", "first"))
    g["overfit_gap"] = g["final_val"] - g["best_val"]
    out.write("## FT final: mean across seeds\n\n"
              "final_val/best_val = shakespeare val· overfit_gap = final−best "
              "(0 σημαίνει «δεν χρειάζεται early stopping»)· retain_enwik8 = "
              "enwik8 val στο τέλος (μικρότερο = λιγότερο forgetting).\n\n"
              + g.round(4).to_markdown() + "\n\n")
    for regime in sorted(fi.regime.unique()):
        for metric, fname in [("val_loss", f"ft_curves_{regime}.png"),
                              ("retain_val", f"ft_retention_{regime}.png")]:
            plt.figure(figsize=(8, 5))
            for arm in sorted(fi[fi.regime == regime].arm.unique()):
                runs = fi[(fi.regime == regime) & (fi.arm == arm)]
                curves = []
                for rid in runs.run_id:
                    log = load_log(rid)
                    if metric in log.columns:
                        v = log.dropna(subset=[metric])
                        if not v.empty:
                            curves.append(v.set_index("step")[metric])
                if curves:
                    m = pd.concat(curves, axis=1).mean(axis=1)
                    plt.plot(m.index, m.values, label=arm, linewidth=2)
            plt.xlabel("step")
            plt.ylabel("shakespeare val loss" if metric == "val_loss" else "enwik8 val loss (retention)")
            plt.title(f"Fine-tuning {regime}" + (" — forgetting" if metric == "retain_val" else ""))
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(PLOTS, fname), dpi=130)
            plt.close()
    out.write("FT curves: `plots/ft_curves_*.png`, forgetting: `plots/ft_retention_*.png`\n\n")


def spectral_analysis(out):
    """Idea 3: cross-correlate jumps in top_sv with future jumps in train loss."""
    recs = []
    for p in glob.glob(os.path.join(RESULTS, "runs", "final_*_muon_s1", "spectral.jsonl")):
        rid = os.path.basename(os.path.dirname(p))
        rows = [json.loads(l) for l in open(p)]
        df = pd.DataFrame(rows)
        if df.empty or "layer" not in df:
            continue
        glob_loss = df[df.layer == "_global"].set_index("step")["train_loss"]
        top = df[df.layer != "_global"].groupby("step")["top_sv"].max()
        steps = sorted(set(glob_loss.index) & set(top.index))
        if len(steps) < 20:
            continue
        dl = glob_loss.loc[steps].diff()
        dt = top.loc[steps].diff()
        for lag in range(0, 6):
            a, b = dt.iloc[1:len(dt) - lag], dl.shift(-lag).iloc[1:len(dl) - lag]
            m = a.notna() & b.notna()
            if m.sum() > 10:
                recs.append({"run": rid, "lag": lag,
                             "corr(d_topSV, d_loss)": float(np.corrcoef(a[m], b[m])[0, 1])})
    if recs:
        t = pd.DataFrame(recs).pivot_table(index="run", columns="lag",
                                           values="corr(d_topSV, d_loss)")
        out.write("## Idea 3: predictive power — corr(Δtop_sv[t], Δtrain_loss[t+lag])\n\n"
                  "Θετική συσχέτιση σε lag>0 = το φάσμα προβλέπει το loss.\n\n"
                  + t.round(3).to_markdown() + "\n\n")

    # what does the controller do: lr_scale heatmap from a muon+ctl run that logged it
    for p in sorted(glob.glob(os.path.join(RESULTS, "runs", "final_*_muonctl_s*", "spectral.jsonl"))):
        rid = os.path.basename(os.path.dirname(p))
        df = pd.DataFrame([json.loads(l) for l in open(p)])
        if "lr_scale" not in df.columns:
            continue
        d = df[df.layer != "_global"]
        piv = d.pivot_table(index="layer", columns="step", values="lr_scale")
        plt.figure(figsize=(10, 6))
        plt.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=0.5, vmax=2.0)
        plt.colorbar(label="controller lr scale")
        plt.yticks(range(len(piv.index)), [i.replace("blocks.", "b") for i in piv.index], fontsize=6)
        plt.xlabel("monitor tick"); plt.title(f"Controller lr scales — {rid}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, "ctl_scales.png"), dpi=130)
        plt.close()
        break

    # stable-rank heatmap for one run
    for p in glob.glob(os.path.join(RESULTS, "runs", "final_*_muon_s1", "spectral.jsonl"))[:2]:
        rid = os.path.basename(os.path.dirname(p))
        df = pd.DataFrame([json.loads(l) for l in open(p)])
        d = df[df.layer != "_global"]
        if d.empty:
            continue
        piv = d.pivot_table(index="layer", columns="step", values="stable_rank")
        plt.figure(figsize=(10, 6))
        plt.imshow(piv.values, aspect="auto", cmap="viridis")
        plt.colorbar(label="stable rank")
        plt.yticks(range(len(piv.index)), [i.replace("blocks.", "b") for i in piv.index], fontsize=6)
        plt.xlabel("monitor tick"); plt.title(f"Momentum stable rank — {rid}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, f"stable_rank_{rid}.png"), dpi=130)
        plt.close()


def main():
    os.makedirs(PLOTS, exist_ok=True)
    df = load_finals()
    if df.empty:
        print("no results yet")
        return
    with open(os.path.join(RESULTS, "report.md"), "w", encoding="utf-8") as out:
        out.write("# Optimizer-lab report\n\nPrimary criterion: **final val loss** at equal "
                  "steps/tokens, best-of-grid lr per optimizer, mean over seeds. "
                  "Secondary: best val, wall clock, tokens/s, stability.\n\n")
        sweep_table(df, out)
        dfinal = final_table(df, out)
        if dfinal is not None:
            welch_table(dfinal, out)
            speed_table(dfinal, out)
            curves_plot(dfinal)
            out.write("Curves: `plots/curves_A.png`, `plots/curves_B.png`\n\n")
        scale_section(df, out)
        ft_section(df, out)
        spectral_analysis(out)
    print(f"report -> {os.path.join(RESULTS, 'report.md')}")


if __name__ == "__main__":
    main()
