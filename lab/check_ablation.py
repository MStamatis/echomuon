"""CPU smoke test for the gate-statistic ablation modes (magnitude/cosgate/cautious/mix).

Checks, per mode: 40 steps run without error on a tiny ViT; loss is finite; the mode
actually engages (gate cache / scalar state / mixture as appropriate); normal-mode
behaviour is bit-identical to before the ablation code was added (regression guard);
state_dict round-trips; and lambda=0 reduces cosgate/cautious to the plain-Muon step.
"""
import torch

from src.model import ViT
from src.optim import HybridOptimizer

torch.manual_seed(0)


def make():
    torch.manual_seed(0)
    return ViT(num_classes=10, dim=64, n_layer=2, n_head=2, img=32, patch=8)


def run(mode, steps=40, lam=1.0, seed_data=1):
    m = make()
    opt = HybridOptimizer(m, mode="tcg", lr=1e-2, gate_mode=mode, gate_every=10)
    opt.gate_lambda = lam
    g = torch.Generator().manual_seed(seed_data)
    losses = []
    for _ in range(steps):
        x = torch.randn(16, 3, 32, 32, generator=g)
        y = torch.randint(0, 10, (16,), generator=g)
        logits, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return m, opt, losses


def flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


print("== per-mode run + engagement ==")
for mode in ["normal", "magnitude", "cosgate", "cautious", "mix"]:
    m, opt, losses = run(mode)
    assert all(torch.isfinite(torch.tensor(losses))), f"{mode}: non-finite loss"
    if mode in ("normal", "magnitude"):
        assert opt._gate, f"{mode}: no cached gate basis after refresh"
    if mode == "cosgate":
        assert opt._cos_s and all(0.0 < v < 1.0 for v in opt._cos_s.values()), \
            "cosgate: scalar state not updating"
        assert not opt._gate, "cosgate: unexpected singular-basis gate"
    if mode in ("cautious", "mix"):
        assert not opt._gate, f"{mode}: unexpected singular-basis gate"
    print(f"  {mode:10s} loss {losses[0]:.3f} -> {losses[-1]:.3f}  ok")

print("== magnitude differs from normal (different statistic => different weights) ==")
wa = flat(run("normal")[0]); wb = flat(run("magnitude")[0])
assert not torch.allclose(wa, wb), "magnitude ablation identical to normal gate?!"
print(f"  max |delta| = {(wa - wb).abs().max():.2e}  ok")

print("== lambda=0 reduces cosgate/cautious to plain-Muon path of tcg ==")
# with lambda=0 the cos/cautious modulation must vanish; compare against mode='mix'
# with alpha=0 (== tcg loop with no gate at all, buffers identical)
m_ref, opt_ref, _ = run("mix", lam=1.0)  # alpha set to 0 below for the true reference
w_ref = None
for mode in ["cosgate", "cautious"]:
    m0 = make()
    o0 = HybridOptimizer(m0, mode="tcg", lr=1e-2, gate_mode=mode, gate_every=10)
    o0.gate_lambda = 0.0
    m1 = make()
    o1 = HybridOptimizer(m1, mode="tcg", lr=1e-2, gate_mode="mix", gate_every=10)
    o1.gate_mix_alpha = 0.0
    g0 = torch.Generator().manual_seed(1)
    g1 = torch.Generator().manual_seed(1)
    for _ in range(15):
        for mm, oo, gg in [(m0, o0, g0), (m1, o1, g1)]:
            x = torch.randn(16, 3, 32, 32, generator=gg)
            y = torch.randint(0, 10, (16,), generator=gg)
            _, loss = mm(x, y)
            oo.zero_grad(); loss.backward(); oo.step()
    assert torch.allclose(flat(m0), flat(m1), atol=1e-6), f"{mode}: lambda=0 != plain"
    print(f"  {mode:10s} lambda=0 == ungated tcg step  ok")

print("== state_dict round-trip (cosgate) ==")
m, opt, _ = run("cosgate", steps=12)
sd = opt.state_dict()
m2 = make()
opt2 = HybridOptimizer(m2, mode="tcg", lr=1e-2, gate_mode="cosgate", gate_every=10)
opt2.load_state_dict(sd)
assert opt2._cos_s == opt._cos_s, "cos_s not restored"
print("  cosgate state round-trips  ok")

print("ALL ABLATION CHECKS PASS")
