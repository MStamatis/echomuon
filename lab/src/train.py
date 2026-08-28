"""Single training run. Writes to <out>/<run_id>/: config.json, log.jsonl, spectral.jsonl, final.json.

Resumable at step level: every --ckpt-every steps an atomic checkpoint (model, optimizer,
RNG states, progress) is written to <run_dir>/ckpt.pt. If the process is killed (crash, PC
restart), re-running the same command resumes from the last checkpoint; a run whose
final.json exists is skipped entirely. The checkpoint is deleted on successful completion.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import load, load_vision
from .model import GPT, Llama, Mamba, ViT
from .optim import HybridOptimizer
from .spectral import SpectralMonitor

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
CIFAR_STD = np.array([0.247, 0.243, 0.261], dtype=np.float32).reshape(3, 1, 1)


def _randaugment_batch(imgs_u8, rng, n_ops=2, m=9):
    """RandAugment(N=2, M=9) on a (B, 3, H, W) uint8 batch via PIL, torchvision-like
    magnitudes. Deterministic through the caller's rng (checkpointed -> resume-safe)."""
    from PIL import Image, ImageEnhance, ImageOps
    f = m / 31.0
    def _affine(im, a, b, c, d, e, g):
        return im.transform(im.size, Image.AFFINE, (a, b, c, d, e, g), resample=Image.BILINEAR)
    ops = [
        lambda im, s: im,
        lambda im, s: ImageOps.autocontrast(im),
        lambda im, s: ImageOps.equalize(im),
        lambda im, s: im.rotate(s * 30.0 * f, resample=Image.BILINEAR),
        lambda im, s: ImageOps.posterize(im, max(1, int(8 - 4 * f))),
        lambda im, s: ImageOps.solarize(im, int(255 * (1 - f))),
        lambda im, s: ImageEnhance.Color(im).enhance(1 + s * 0.9 * f),
        lambda im, s: ImageEnhance.Contrast(im).enhance(1 + s * 0.9 * f),
        lambda im, s: ImageEnhance.Brightness(im).enhance(1 + s * 0.9 * f),
        lambda im, s: ImageEnhance.Sharpness(im).enhance(1 + s * 0.9 * f),
        lambda im, s: _affine(im, 1, s * 0.3 * f, 0, 0, 1, 0),          # shear x
        lambda im, s: _affine(im, 1, 0, 0, s * 0.3 * f, 1, 0),          # shear y
        lambda im, s: _affine(im, 1, 0, s * f * im.size[0] / 3, 0, 1, 0),  # translate x
        lambda im, s: _affine(im, 1, 0, 0, 0, 1, s * f * im.size[1] / 3),  # translate y
    ]
    out = np.empty_like(imgs_u8)
    picks = rng.integers(0, len(ops), (len(imgs_u8), n_ops))
    signs = rng.choice([-1.0, 1.0], (len(imgs_u8), n_ops))
    for i in range(len(imgs_u8)):
        im = Image.fromarray(imgs_u8[i].transpose(1, 2, 0))
        for k in range(n_ops):
            im = ops[picks[i, k]](im, signs[i, k])
        out[i] = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)
    return out


def get_vision_batch(x_arr, y_arr, batch, device, rng, augment, spec=None, randaug=False):
    img, mean, std = spec if spec else (32, CIFAR_MEAN, CIFAR_STD)
    pad = img // 8
    idx = rng.integers(0, len(y_arr), batch)
    raw = np.asarray(x_arr[idx]).reshape(batch, 3, img, img)
    if augment and randaug:
        raw = _randaugment_batch(raw, rng)
    imgs = raw.astype(np.float32) / 255.0
    if augment:
        padded = np.pad(imgs, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")
        out = np.empty_like(imgs)
        offs = rng.integers(0, 2 * pad + 1, (batch, 2))
        flips = rng.random(batch) < 0.5
        for i in range(batch):
            r, c = offs[i]
            crop = padded[i, :, r:r + img, c:c + img]
            out[i] = crop[:, :, ::-1] if flips[i] else crop
        imgs = out
    imgs = (imgs - mean) / std
    x = torch.from_numpy(imgs).to(device, non_blocking=True)
    y = torch.from_numpy(np.asarray(y_arr[idx]).astype(np.int64)).to(device, non_blocking=True)
    return x, y, idx


@torch.no_grad()
def evaluate_vision(model, x_arr, y_arr, args, device):
    model.eval()
    rng = np.random.default_rng(1234)  # same eval batches every time
    losses, correct, total = [], 0, 0
    for _ in range(args.eval_iters):
        x, y, _ = get_vision_batch(x_arr, y_arr, args.batch, device, rng, augment=False,
                                   spec=getattr(args, "vspec", None))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, loss = model(x, y)
        losses.append(loss.item())
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    model.train()
    return float(np.mean(losses)), correct / total


def get_batch(data, batch, block, device, generator):
    ix = torch.randint(len(data) - block - 1, (batch,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), ix


@torch.no_grad()
def evaluate(model, data, args, device):
    model.eval()
    g = torch.Generator().manual_seed(1234)  # same eval batches for every run/step
    losses = []
    for _ in range(args.eval_iters):
        x, y, _ = get_batch(data, args.batch, args.block, device, g)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def truncate_jsonl(path, max_step):
    """Drop log lines from steps after the checkpoint we are resuming from."""
    if not os.path.exists(path):
        return
    kept = []
    with open(path) as f:
        for line in f:
            try:
                if json.loads(line).get("step", 0) <= max_step:
                    kept.append(line)
            except json.JSONDecodeError:
                pass
    with open(path, "w") as f:
        f.writelines(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["adamw", "muon", "shrunk", "tcg"], required=True)
    ap.add_argument("--slow-beta", type=float, default=0.99)
    ap.add_argument("--gate-every", type=int, default=25)
    ap.add_argument("--gate-floor", type=float, default=0.1)
    ap.add_argument("--gate-mode",
                    choices=["normal", "inverse", "shuffled", "novelty", "coherence",
                             "amplify", "magnitude", "cosgate", "cautious", "mix",
                             "scalar"],
                    default="normal")
    ap.add_argument("--gate-quantile", type=float, default=0.5)
    ap.add_argument("--gate-block", type=int, default=1)
    ap.add_argument("--gate-stage", choices=["post", "pre"], default="post")
    ap.add_argument("--no-augment", action="store_true",
                    help="vision: disable train-time crop/flip augmentation")
    ap.add_argument("--randaugment", action="store_true",
                    help="vision strong recipe: RandAugment(2, 9) on train batches")
    ap.add_argument("--mixup", type=float, default=0.0,
                    help="vision strong recipe: mixup Beta(alpha, alpha); 0 disables")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="vision strong recipe: label smoothing on the train loss")
    ap.add_argument("--auto-gate", action="store_true",
                    help="EchoMuon controller: scale the gate by measured overfitting — lambda = "
                         "clip(gap / (5%% of probe loss), 0, 1), gap = held-out-train "
                         "probe loss minus seen-train EMA")
    ap.add_argument("--probe-frac", type=float, default=0.02)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--auto-version", type=int, default=1,
                    help="1: held-out-probe vs train-EMA gap; 2: memorization gap "
                         "(fresh samples vs re-evaluated recently-seen batches)")
    ap.add_argument("--ring-per-probe", action="store_true",
                    help="auto2: append to the seen-batch ring once per probe interval "
                         "instead of every step — retention horizon becomes 4-7 probe "
                         "intervals (400-700 steps at probe_every=100) instead of 4-7 steps")
    ap.add_argument("--fixed-lambda", type=float, default=-1.0,
                    help=">=0: pin the gate strength lambda to this value for the whole "
                         "run (no controller); the lambda-ladder ablation arm")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="vision: hold out this fraction of the (possibly noisy) train "
                         "set as a validation split; eval/selection metrics then use it "
                         "instead of the test set (sweep-only protocol)")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--aux-lr", type=float, default=2e-3)
    ap.add_argument("--muon-wd", type=float, default=0.0,
                    help="decoupled weight decay on the Muon-class matrix params")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--grad-noise", type=float, default=0.0,
                    help="Gaussian grad noise, sigma = value * per-tensor grad RMS (low-SNR regime)")
    ap.add_argument("--spectral-lr", action="store_true", help="Idea-3 per-layer lr controller")
    ap.add_argument("--ctl-mode", choices=["srank", "inverse", "shuffled"], default="srank",
                    help="controller variant (ablations); only used with --spectral-lr")
    ap.add_argument("--ctl-signals", default="rank",
                    help="comma list of spectral signals: rank,conf,valve (spectral-ctl = all)")
    ap.add_argument("--monitor-every", type=int, default=25)
    ap.add_argument("--no-monitor", action="store_true")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lr-schedule", choices=["const", "cosine"], default="const",
                    help="cosine decays the matrix lr to 0 over the run (control for the "
                         "implicit annealing that conf2 provides)")
    ap.add_argument("--lr-spike", default="",
                    help="fault injection 'start:len:mult' — multiply matrix lr by mult "
                         "for len steps starting at start (valve stress test)")
    ap.add_argument("--shrink-mix", type=float, default=0.0)
    ap.add_argument("--task", choices=["lm", "vision"], default="lm")
    ap.add_argument("--arch", choices=["gpt", "llama", "mamba"], default="gpt",
                    help="LM architecture: gpt (LayerNorm/GELU/learned pos), llama "
                         "(RMSNorm/RoPE/SwiGLU), mamba (Mamba-2-style SSM, no attention)")
    ap.add_argument("--dataset",
                    choices=["shakespeare", "shakespeare_bytes", "enwik8", "enwik8p10",
                             "fineweb",
                             "cifar10", "cifar10n10", "cifar10n20", "cifar10n40",
                             "cifar100", "cifar100n10", "cifar100n20", "cifar100n40",
                             "tinyimagenet", "tinyimagenetn20"],
                    default="shakespeare")
    ap.add_argument("--extra-val", default="",
                    help="second dataset whose val split is also evaluated (retention metric)")
    ap.add_argument("--init-from", default="", help="checkpoint to initialize weights from")
    ap.add_argument("--save-checkpoint", default="", help="save final model weights here")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    ap.add_argument("--out", default=os.environ.get("RESULTS_DIR", "results"))
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    run_dir = os.path.join(args.out, "runs", args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    final_path = os.path.join(run_dir, "final.json")
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    log_path = os.path.join(run_dir, "log.jsonl")
    spectral_path = os.path.join(run_dir, "spectral.jsonl")
    if os.path.exists(final_path):
        print(f"[{args.run_id}] final.json exists, skipping")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        print(f"[{args.run_id}] device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[{args.run_id}] device: CPU (this will be slow)")

    extra_val_data = None
    probe_lm = probe_x = probe_y = None
    if args.task == "vision":
        train_x, train_y, test_x, test_y, vmeta = load_vision(args.data_dir, args.dataset)
        vimg = vmeta["img_size"]
        args.vspec = (vimg,
                      np.array(vmeta["mean"], dtype=np.float32).reshape(3, 1, 1),
                      np.array(vmeta["std"], dtype=np.float32).reshape(3, 1, 1))
        eval_x, eval_y = test_x, test_y
        if args.val_frac > 0:  # sweep-only: select on a held-out train slice, not test
            # Shuffle before slicing: Tiny-ImageNet is decoded in wnid order, so a
            # contiguous tail is class-disjoint from the head (the val slice would
            # hold only classes the train slice never shows). Fixed seed, independent
            # of --seed, so every lr in a sweep selects on the identical split.
            perm = np.random.default_rng(1234).permutation(len(train_y))
            train_x, train_y = train_x[perm], train_y[perm]
            cut = int(len(train_y) * (1 - args.val_frac))
            eval_x, eval_y = train_x[cut:], train_y[cut:]
            train_x, train_y = train_x[:cut], train_y[:cut]
        if args.auto_gate and args.auto_version == 1:  # v1: held-out TRAIN probe slice
            cut = int(len(train_y) * (1 - args.probe_frac))
            probe_x, probe_y = train_x[cut:], train_y[cut:]
            train_x, train_y = train_x[:cut], train_y[:cut]
        model = ViT(vmeta["num_classes"], vimg, vimg // 8,
                    args.n_layer, args.n_head, args.dim).to(device)
    else:
        train_data, val_data, vocab = load(args.data_dir, args.dataset)
        if args.auto_gate and args.auto_version == 1:
            cut = int(len(train_data) * (1 - args.probe_frac))
            probe_lm = train_data[cut:]
            train_data = train_data[:cut]
        if args.extra_val:
            _, extra_val_data, _ = load(args.data_dir, args.extra_val)
        if args.arch == "llama":
            model = Llama(vocab, args.block, args.n_layer, args.n_head, args.dim).to(device)
        elif args.arch == "mamba":
            model = Mamba(vocab, args.block, args.n_layer, args.dim).to(device)
        else:
            model = GPT(vocab, args.block, args.n_layer, args.n_head, args.dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = HybridOptimizer(model, args.optimizer, lr=args.lr, aux_lr=args.aux_lr,
                          wd=args.muon_wd, slow_beta=args.slow_beta,
                          gate_every=args.gate_every, gate_floor=args.gate_floor,
                          gate_mode=args.gate_mode, gate_quantile=args.gate_quantile,
                          seed=args.seed)
    opt.gate_block = args.gate_block
    opt.gate_stage = args.gate_stage
    if args.fixed_lambda >= 0:
        opt.gate_lambda = args.fixed_lambda
    opt.attach_names(model)
    g_train = torch.Generator().manual_seed(args.seed)
    noise_gen = torch.Generator(device=device).manual_seed(args.seed + 9999)
    aug_rng = np.random.default_rng(args.seed)  # vision batches + augmentation

    start_step, best_val, wall_prev = 0, float("inf"), 0.0
    if os.path.exists(ckpt_path):
        # map to CPU: RNG states must stay ByteTensors on CPU; model/optimizer
        # load_state_dict move their tensors onto the right device themselves
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        g_train.set_state(ck["g_train"])
        noise_gen.set_state(ck["noise_gen"])
        if ck.get("aug_rng") is not None:
            aug_rng.bit_generator.state = ck["aug_rng"]
        torch.set_rng_state(ck["torch_rng"])
        if device == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        start_step, best_val, wall_prev = ck["step"], ck["best_val"], ck["wall_s"]
        truncate_jsonl(log_path, start_step)
        truncate_jsonl(spectral_path, start_step)
        print(f"[{args.run_id}] RESUMING from step {start_step}")
    elif args.init_from:
        ck = torch.load(args.init_from, map_location="cpu")
        model.load_state_dict(ck["model"])
        print(f"[{args.run_id}] initialized from {args.init_from}")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        cfg = {k: v for k, v in vars(args).items() if k != "vspec"}
        json.dump({**cfg, "n_params": n_params, "device": device}, f, indent=2)

    monitor = None
    if not args.no_monitor:
        monitor = SpectralMonitor(opt, args.monitor_every, spectral_path,
                                  controller=args.spectral_lr, ctl_mode=args.ctl_mode,
                                  signals=tuple(args.ctl_signals.split(",")), seed=args.seed)
    log = open(log_path, "a", buffering=1)
    t0 = time.time()

    def save_ckpt(step):
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "g_train": g_train.get_state(), "noise_gen": noise_gen.get_state(),
                    "aug_rng": aug_rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                    "step": step, "best_val": best_val,
                    "wall_s": wall_prev + time.time() - t0}, tmp)
        os.replace(tmp, ckpt_path)

    spike = None
    if args.lr_spike:
        s0, ln, mult = (float(v) for v in args.lr_spike.split(":"))
        spike = (int(s0), int(ln), mult)

    vl = None
    retain = None
    acc = None
    best_acc = 0.0
    ema_train = None
    lam_hist = []
    seen_ring = []
    last_batch_ref = None
    if args.auto_gate and args.auto_version == 2:
        # memorization gap: fresh random samples vs re-evaluated recently-seen batches,
        # both measured on the same model at the same moment (corpus drift cancels)
        g_fresh = torch.Generator().manual_seed(args.seed + 4242)
        fresh_np = np.random.default_rng(args.seed + 4242)
        if args.task == "vision":
            def refetch(idx):
                idx = np.asarray(idx)
                vimg, vmean, vstd = args.vspec
                imgs = np.asarray(train_x[np.sort(idx)]).reshape(len(idx), 3, vimg, vimg)
                imgs = (imgs.astype(np.float32) / 255.0 - vmean) / vstd
                xb = torch.from_numpy(imgs).to(device)
                yb = torch.from_numpy(
                    np.asarray(train_y[np.sort(idx)]).astype(np.int64)).to(device)
                return xb, yb

            def fresh_batch():
                return refetch(fresh_np.integers(0, len(train_y), args.batch))
        else:
            def refetch(ix):
                if hasattr(ix, "tolist"):
                    ix = ix.tolist()
                x = torch.stack([torch.from_numpy(
                    train_data[i:i + args.block].astype(np.int64)) for i in ix])
                y = torch.stack([torch.from_numpy(
                    train_data[i + 1:i + 1 + args.block].astype(np.int64)) for i in ix])
                return x.to(device), y.to(device)

            def fresh_batch():
                ix = torch.randint(len(train_data) - args.block - 1, (args.batch,),
                                   generator=g_fresh)
                return refetch(ix.tolist())
    import math as _math
    for step in range(start_step + 1, args.steps + 1):
        lr_t = args.lr
        if args.lr_schedule == "cosine":
            lr_t = args.lr * 0.5 * (1 + _math.cos(_math.pi * step / args.steps))
        if spike:
            s0, ln, mult = spike
            lr_t *= mult if s0 <= step < s0 + ln else 1.0
        opt.lr = lr_t
        if args.task == "vision":
            x, y, last_batch_ref = get_vision_batch(train_x, train_y, args.batch, device,
                                                    aug_rng, augment=not args.no_augment,
                                                    spec=args.vspec,
                                                    randaug=args.randaugment)
        else:
            x, y, last_batch_ref = get_batch(train_data, args.batch, args.block, device,
                                             g_train)
        if args.task == "vision" and (args.mixup > 0 or args.label_smoothing > 0):
            # strong recipe: loss computed outside the model so mixup and smoothing
            # touch only the TRAIN objective (eval and the memorization probes stay
            # plain CE, identical across arms)
            y2 = None
            lam_mix = 1.0
            if args.mixup > 0:
                lam_mix = float(aug_rng.beta(args.mixup, args.mixup))
                perm = torch.from_numpy(aug_rng.permutation(len(y))).to(device)
                x = lam_mix * x + (1 - lam_mix) * x[perm]
                y2 = y[perm]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                logits, _ = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            if y2 is not None:
                loss = lam_mix * loss + (1 - lam_mix) * F.cross_entropy(
                    logits, y2, label_smoothing=args.label_smoothing)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        if args.grad_noise > 0:
            for p in model.parameters():
                if p.grad is not None:
                    rms = p.grad.float().pow(2).mean().sqrt()
                    p.grad.add_(torch.randn(p.grad.shape, generator=noise_gen,
                                            device=device, dtype=torch.float32).to(p.grad.dtype),
                                alpha=float(rms) * args.grad_noise)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        train_loss = loss.item()
        if args.auto_gate and args.auto_version == 2:
            if (not args.ring_per_probe) or step % args.probe_every == 0:
                seen_ring.append(last_batch_ref)
                if len(seen_ring) > 8:
                    seen_ring.pop(0)
            if step % args.probe_every == 0 and len(seen_ring) >= 8:
                model.eval()
                with torch.no_grad():
                    reseen, fresh = [], []
                    for ref in seen_ring[:4]:  # oldest 4 of the ring (seen 400-800 steps ago)
                        xb, yb = refetch(ref)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                            enabled=device == "cuda"):
                            _, l = model(xb, yb)
                        reseen.append(l.item())
                    for _ in range(4):
                        xb, yb = fresh_batch()
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                            enabled=device == "cuda"):
                            _, l = model(xb, yb)
                        fresh.append(l.item())
                model.train()
                gap = float(np.mean(fresh) - np.mean(reseen))
                lam_raw = float(np.clip(gap / (0.02 * max(np.mean(fresh), 1e-6)), 0.0, 1.0))
                lam = lam_raw if not lam_hist else 0.7 * lam_hist[-1] + 0.3 * lam_raw
                opt.gate_lambda = lam
                lam_hist.append(lam)
                log.write(json.dumps({"step": step, "fresh_loss": float(np.mean(fresh)),
                                      "reseen_loss": float(np.mean(reseen)),
                                      "gap": gap, "gate_lambda": lam}) + "\n")
        elif args.auto_gate:
            ema_train = train_loss if ema_train is None else 0.95 * ema_train + 0.05 * train_loss
            if step % args.probe_every == 0:
                saved_iters = args.eval_iters
                args.eval_iters = 4
                if args.task == "vision":
                    probe_loss, _ = evaluate_vision(model, probe_x, probe_y, args, device)
                else:
                    probe_loss = evaluate(model, probe_lm, args, device)
                args.eval_iters = saved_iters
                gap = probe_loss - ema_train
                lam = float(np.clip(gap / (0.05 * max(probe_loss, 1e-6)), 0.0, 1.0))
                opt.gate_lambda = lam
                lam_hist.append(lam)
                log.write(json.dumps({"step": step, "probe_loss": probe_loss,
                                      "gap": gap, "gate_lambda": lam}) + "\n")
        if step % 10 == 0 or step == 1:
            log.write(json.dumps({"step": step, "train_loss": train_loss,
                                  "t": wall_prev + time.time() - t0}) + "\n")
        if monitor:
            monitor.maybe_log(step, {"train_loss": train_loss})
        if step % args.eval_every == 0 or step == args.steps:
            rec = {"step": step, "t": wall_prev + time.time() - t0}
            if args.task == "vision":
                vl, acc = evaluate_vision(model, eval_x, eval_y, args, device)
                best_acc = max(best_acc, acc)
                rec["acc"] = acc
                msg_extra = f" acc {acc:.4f}"
            else:
                vl = evaluate(model, val_data, args, device)
                msg_extra = ""
            best_val = min(best_val, vl)
            rec["val_loss"] = vl
            msg = (f"[{args.run_id}] step {step}/{args.steps} train {train_loss:.4f} "
                   f"val {vl:.4f}" + msg_extra)
            if extra_val_data is not None:
                retain = evaluate(model, extra_val_data, args, device)
                rec["retain_val"] = retain
                msg += f" retain {retain:.4f}"
            log.write(json.dumps(rec) + "\n")
            print(msg)
        if step % args.ckpt_every == 0 and step < args.steps:
            save_ckpt(step)

    if vl is None:  # resumed exactly at the end without re-running the loop
        if args.task == "vision":
            vl, acc = evaluate_vision(model, eval_x, eval_y, args, device)
            best_acc = max(best_acc, acc)
        else:
            vl = evaluate(model, val_data, args, device)
            if extra_val_data is not None:
                retain = evaluate(model, extra_val_data, args, device)
        best_val = min(best_val, vl)

    wall = wall_prev + time.time() - t0
    if args.save_checkpoint:
        os.makedirs(os.path.dirname(args.save_checkpoint), exist_ok=True)
        torch.save({"model": model.state_dict(), "config": vars(args)}, args.save_checkpoint)
        print(f"[{args.run_id}] saved model checkpoint -> {args.save_checkpoint}")
    with open(final_path, "w") as f:
        json.dump({"run_id": args.run_id, "optimizer": args.optimizer, "lr": args.lr,
                   "arch": args.arch,
                   "dataset": args.dataset, "seed": args.seed, "grad_noise": args.grad_noise,
                   "batch": args.batch, "muon_wd": args.muon_wd,
                   "spectral_lr": args.spectral_lr,
                   "ctl_mode": args.ctl_mode if args.spectral_lr else None,
                   "ctl_signals": args.ctl_signals if args.spectral_lr else None,
                   "steps": args.steps,
                   "n_params": n_params, "task": args.task,
                   "lr_schedule": args.lr_schedule, "gate_mode": args.gate_mode,
                   "gate_quantile": args.gate_quantile,
                   "gate_block": args.gate_block, "gate_stage": args.gate_stage,
                   "slow_beta": args.slow_beta,
                   "gate_every": args.gate_every, "probe_every": args.probe_every,
                   "auto_gate": args.auto_gate, "auto_version": args.auto_version,
                   "ring_per_probe": args.ring_per_probe,
                   "fixed_lambda": args.fixed_lambda if args.fixed_lambda >= 0 else None,
                   "val_frac": args.val_frac,
                   "mean_lambda": float(np.mean(lam_hist)) if lam_hist else None,
                   "final_val": vl, "best_val": best_val, "retain_val": retain,
                   "final_acc": acc, "best_acc": best_acc if args.task == "vision" else None,
                   "wall_s": wall, "tok_per_s": args.steps * args.batch * args.block / wall}, f,
                  indent=2)
    log.close()
    if monitor:
        monitor.close()
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"[{args.run_id}] DONE final_val={vl:.4f} best_val={best_val:.4f} wall={wall:.0f}s")


if __name__ == "__main__":
    main()
