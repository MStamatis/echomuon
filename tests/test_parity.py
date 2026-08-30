"""The shipped optimizer must reproduce the research code bit for bit.

Every number in the paper came out of ``lab/src/optim.py``. Everything a reader can
install comes out of ``src/echomuon/optimizer.py``. Nothing checked that the two agree:
``lab/v2_bitcheck.py`` compares the lab against an older copy of itself, not against the
package. They did in fact differ, in the order the gate basis is refreshed relative to
the momentum update, and the difference went unnoticed for two releases.

This test pins them together. If it fails, either the package drifted from the paper or
the lab did, and any v2 comparison run across the two is meaningless until it passes.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from echomuon import EchoMuon

LR = 0.02
STEPS = 30
GATE_EVERY = 5
SHAPES = [(32, 64), (64, 32), (48, 48)]     # wide, tall and square: all transpose paths


def _load_lab():
    """Import lab/src/optim.py under a private package name.

    Loading it as plain ``src.optim`` would race with this repo's own ``src/``
    directory on sys.path, so give it a namespace of its own and let the module's
    ``from .rmt import ...`` resolve against that.
    """
    for base in [Path(__file__).resolve().parent] + list(Path(__file__).resolve().parents):
        cand = base / "lab" / "src" / "optim.py"
        if cand.exists():
            pkg = types.ModuleType("_labsrc")
            pkg.__path__ = [str(cand.parent)]
            sys.modules["_labsrc"] = pkg
            spec = importlib.util.spec_from_file_location("_labsrc.optim", cand)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_labsrc.optim"] = mod
            spec.loader.exec_module(mod)
            return mod
    return None


class _FakeModel(torch.nn.Module):
    """Minimal stand-in for the lab's model protocol.

    ``dummy`` exists only because the lab optimizer always builds an AdamW for the
    non-matrix parameters and AdamW rejects an empty list. It never receives a
    gradient, so AdamW skips it and the comparison stays clean.
    """

    def __init__(self, mats):
        super().__init__()
        self.mats = torch.nn.ParameterList([torch.nn.Parameter(m.clone()) for m in mats])
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def hidden_matrix_names(self):
        return [n for n, p in self.named_parameters() if p.ndim == 2]


def _grad_stream(seed=0):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield [torch.randn(*s, generator=g) for s in SHAPES]


@pytest.mark.parametrize("gate_lambda", [1.0, 0.5])
def test_package_matches_lab_bit_for_bit(gate_lambda):
    lab = _load_lab()
    if lab is None:
        pytest.skip("lab/src/optim.py not present (running outside the research repo)")

    init = [torch.randn(*s, generator=torch.Generator().manual_seed(7)) for s in SHAPES]

    model = _FakeModel(init)
    lab_opt = lab.HybridOptimizer(
        model, mode="tcg", lr=LR, momentum=0.95, nesterov=True, wd=0.0,
        ns_steps=5, slow_beta=0.99, gate_every=GATE_EVERY, gate_floor=0.1,
        gate_mode="normal")
    lab_opt.gate_lambda = gate_lambda
    lab_params = [p for n, p in lab_opt.matrix_params]

    pkg_params = [torch.nn.Parameter(m.clone()) for m in init]
    pkg_opt = EchoMuon(pkg_params, lr=LR, momentum=0.95, slow_beta=0.99,
                       nesterov=True, weight_decay=0.0, ns_steps=5,
                       gate_every=GATE_EVERY, gate_floor=0.1,
                       gate_lambda=gate_lambda)

    stream = _grad_stream()
    for step in range(1, STEPS + 1):
        grads = next(stream)
        for p, g in zip(lab_params, grads):
            p.grad = g.clone()
        for p, g in zip(pkg_params, grads):
            p.grad = g.clone()
        lab_opt.step()
        pkg_opt.step()
        for i, (a, b) in enumerate(zip(lab_params, pkg_params)):
            assert torch.equal(a, b), (
                "diverged at step %d, matrix %d %s: max |lab - pkg| = %.3e"
                % (step, i, tuple(a.shape), float((a - b).abs().max())))

    # guard against a vacuous pass: the run has to have actually moved and gated
    assert not torch.equal(lab_params[0], init[0]), "no update happened at all"
    gates = [pkg_opt.state[p]["gate_g"] for p in pkg_params if "gate_g" in pkg_opt.state[p]]
    assert gates, "the gate never refreshed, so this proves nothing about the gate"
    assert min(float(g.min()) for g in gates) < 1.0, "the gate was inert this run"


def test_gate_every_is_read_per_group():
    """A second group with its own cadence must not silently inherit group 0's."""
    a = torch.nn.Parameter(torch.randn(16, 32, generator=torch.Generator().manual_seed(1)))
    b = torch.nn.Parameter(torch.randn(16, 32, generator=torch.Generator().manual_seed(2)))
    opt = EchoMuon([{"params": [a], "gate_every": 2},
                    {"params": [b], "gate_every": 1000}], lr=LR)
    g = torch.Generator().manual_seed(3)
    for _ in range(4):
        a.grad = torch.randn(16, 32, generator=g)
        b.grad = torch.randn(16, 32, generator=g)
        opt.step()
    assert "gate_g" in opt.state[a], "group 0 (gate_every=2) should have refreshed"
    assert "gate_g" not in opt.state[b], "group 1 (gate_every=1000) should not have"
