"""Phase-2 sanity checks (CPU, no GPU needed):
1. _ssd chunk-parallel scan == naive sequential SSM recurrence
2. Llama / Mamba forward+backward shapes, param routing (Muon vs AdamW sets)
Run: python check_phase2.py
"""
import torch

from src.model import Llama, Mamba, _ssd


def test_ssd_correctness():
    torch.manual_seed(0)
    b, l, h, p, n, chunk = 2, 128, 3, 5, 7, 32
    x = torch.randn(b, l, h, p)
    A = -torch.rand(b, l, h) * 0.5
    B = torch.randn(b, l, n)
    C = torch.randn(b, l, n)
    Y = _ssd(x, A, B, C, chunk)
    # naive recurrence: state_t = exp(A_t) * state_{t-1} + B_t x_t ; y_t = C_t . state_t
    state = torch.zeros(b, h, p, n)
    Yn = torch.zeros(b, l, h, p)
    for t in range(l):
        state = torch.exp(A[:, t])[:, :, None, None] * state \
            + torch.einsum("bn,bhp->bhpn", B[:, t], x[:, t])
        Yn[:, t] = torch.einsum("bn,bhpn->bhp", C[:, t], state)
    err = (Y - Yn).abs().max().item()
    rel = err / Yn.abs().max().item()
    assert rel < 1e-4, f"SSD mismatch: max abs err {err}, rel {rel}"
    print(f"SSD OK: max abs err {err:.2e} (rel {rel:.2e})")


def test_models():
    for name, model in [("llama", Llama(256, 64, 2, 2, 64)),
                        ("mamba", Mamba(256, 64, 2, 64))]:
        idx = torch.randint(0, 256, (4, 64))
        tgt = torch.randint(0, 256, (4, 64))
        logits, loss = model(idx, tgt)
        assert logits.shape == (4, 64, 256)
        loss.backward()
        assert all(pp.grad is not None for pp in model.parameters()), f"{name}: missing grads"
        hidden = set(model.hidden_matrix_names())
        aux = [nm for nm, pp in model.named_parameters() if nm not in hidden]
        assert all(pp.ndim == 2 for nm, pp in model.named_parameters() if nm in hidden)
        assert not any(nm.startswith("blocks.") and pp.ndim == 2 and "conv" not in nm
                       for nm, pp in model.named_parameters() if nm not in hidden)
        n_hidden = sum(pp.numel() for nm, pp in model.named_parameters() if nm in hidden)
        print(f"{name} OK: loss {loss.item():.3f}, {len(hidden)} Muon matrices "
              f"({n_hidden} params), {len(aux)} AdamW tensors")


if __name__ == "__main__":
    test_ssd_correctness()
    test_models()
    print("ALL PHASE-2 CHECKS PASSED")
