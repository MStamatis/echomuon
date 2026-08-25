"""CPU tests for the echomuon package. Run: python tests/test_echomuon.py (or pytest)."""
import copy
import sys

import torch

from echomuon import EchoMuon, MemorizationGapController


def make_model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(32, 64, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10, bias=False),
    )


def data(seed=1, n=60):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        x = torch.randn(16, 32, generator=g)
        y = torch.randint(0, 10, (16,), generator=g)
        yield x, y


def run(model, opt, n=60, lam=None):
    losses = []
    for x, y in data(n=n):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        if lam is not None:
            opt.gate_lambda = lam
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_trains():
    # a memorizable task: cycle over the same 4 fixed batches
    m = make_model()
    opt = EchoMuon(m.parameters(), lr=0.02, gate_every=10)
    batches = list(data(n=4))
    losses = []
    for i in range(80):
        x, y = batches[i % 4]
        loss = torch.nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    assert all(torch.isfinite(torch.tensor(losses)))
    assert sum(losses[-8:]) < sum(losses[:8]), "loss did not decrease"


def test_lambda_zero_is_plain_muon():
    # gate_lambda=0 must be bit-identical to never refreshing the gate at all
    m0 = make_model()
    o0 = EchoMuon(m0.parameters(), lr=0.02, gate_every=10, gate_lambda=0.0)
    m1 = make_model()
    o1 = EchoMuon(m1.parameters(), lr=0.02, gate_every=10 ** 9)  # gate never built
    run(m0, o0, n=30)
    run(m1, o1, n=30)
    assert torch.allclose(flat(m0), flat(m1), atol=0), "lambda=0 != plain Muon"


def test_gate_changes_trajectory():
    m0 = make_model()
    o0 = EchoMuon(m0.parameters(), lr=0.02, gate_every=10, gate_lambda=1.0)
    m1 = make_model()
    o1 = EchoMuon(m1.parameters(), lr=0.02, gate_every=10, gate_lambda=0.0)
    run(m0, o0, n=30)
    run(m1, o1, n=30)
    assert not torch.allclose(flat(m0), flat(m1)), "gate had no effect"


def test_rejects_non_2d():
    lin = torch.nn.Linear(8, 8, bias=True)  # bias is 1-D
    try:
        EchoMuon(lin.parameters(), lr=0.02)
    except ValueError:
        return
    raise AssertionError("non-2D parameter was accepted")


def test_state_dict_roundtrip():
    m = make_model()
    opt = EchoMuon(m.parameters(), lr=0.02, gate_every=10)
    run(m, opt, n=25)
    sd_model = copy.deepcopy(m.state_dict())
    sd_opt = copy.deepcopy(opt.state_dict())
    # continue 10 more steps -> reference
    ref = copy.deepcopy(m)
    ref_opt = EchoMuon(ref.parameters(), lr=0.02, gate_every=10)
    ref.load_state_dict(sd_model)
    ref_opt.load_state_dict(copy.deepcopy(sd_opt))
    # note: generators differ, so drive both with the SAME fresh data
    g = torch.Generator().manual_seed(99)
    batches = [(torch.randn(16, 32, generator=g), torch.randint(0, 10, (16,), generator=g))
               for _ in range(10)]
    for x, y in batches:
        loss = torch.nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    m2 = ref
    for x, y in batches:
        loss = torch.nn.functional.cross_entropy(m2(x), y)
        ref_opt.zero_grad(); loss.backward(); ref_opt.step()
    assert torch.allclose(flat(m), flat(m2), atol=1e-6), "resume diverged"


def test_controller():
    m = make_model()
    opt = EchoMuon(m.parameters(), lr=0.02, gate_every=10)
    ctl = MemorizationGapController(opt, probe_every=5, ema=0.0)
    assert not ctl.due(3) and ctl.due(5)
    lam = ctl.update(fresh_loss=2.0, reseen_loss=2.0)   # no gap -> 0
    assert lam == 0.0 and opt.gate_lambda == 0.0
    lam = ctl.update(fresh_loss=2.0, reseen_loss=1.0)   # huge gap -> clip 1
    assert lam == 1.0 and opt.gate_lambda == 1.0
    lam = ctl.update(fresh_loss=2.0, reseen_loss=2.0 - 0.02)  # gap = 0.5*target
    assert abs(lam - 0.5) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  {fn.__name__} ... ok")
    print(f"ALL {len(fns)} TESTS PASS")
    sys.exit(0)
