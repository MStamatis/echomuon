"""Minimal GPT (nanoGPT-style), bias-free, untied embeddings so optimizer roles stay clean:
hidden Linear weights -> Muon-class optimizers; embeddings / lm_head / norms -> AdamW."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, dim: int, n_head: int, causal: bool = True):
        super().__init__()
        self.n_head = n_head
        self.causal = causal
        self.ln1 = nn.LayerNorm(dim, bias=False)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ln2 = nn.LayerNorm(dim, bias=False)
        self.fc = nn.Linear(dim, 4 * dim, bias=False)
        self.fc_proj = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        x = x + self.fc_proj(F.gelu(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, dim: int):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(block_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(dim, bias=False)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def hidden_matrix_names(self):
        """Names of 2-D hidden weights (the Muon-managed set)."""
        return [n for n, p in self.named_parameters() if p.ndim == 2 and n.startswith("blocks.")]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * self.weight.float()).to(x.dtype)


def _rope_cache(head_dim: int, max_t: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(max_t).float(), inv)  # (T, head_dim/2)
    return torch.cos(ang), torch.sin(ang)


def _apply_rope(x, cos, sin):
    """x: (B, H, T, head_dim), GPT-NeoX half-split rotation."""
    T = x.size(2)
    c, s = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class LlamaBlock(nn.Module):
    def __init__(self, dim: int, n_head: int, hidden: int):
        super().__init__()
        self.n_head = n_head
        self.ln1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ln2 = RMSNorm(dim)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        h = self.ln2(x)
        x = x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return x


class Llama(nn.Module):
    """LLaMA-style decoder (RMSNorm pre-norm, RoPE, SwiGLU, bias-free, untied embeddings)
    — the modern-standard LM parameterization. Same optimizer split as GPT: 2-D block
    weights -> Muon-class, embeddings / lm_head / norm scales -> AdamW."""

    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, dim: int):
        super().__init__()
        self.block_size = block_size
        hidden = 64 * round(dim * 8 / 3 / 64)  # SwiGLU width, multiple of 64
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([LlamaBlock(dim, n_head, hidden) for _ in range(n_layer)])
        self.ln_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        cos, sin = _rope_cache(dim // n_head, block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def hidden_matrix_names(self):
        return [n for n, p in self.named_parameters() if p.ndim == 2 and n.startswith("blocks.")]


def _segsum(x):
    """(..., T) -> (..., T, T) lower-triangular segment sums; -inf above the diagonal."""
    T = x.size(-1)
    cs = x.cumsum(-1)
    ss = cs[..., :, None] - cs[..., None, :]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    return ss.masked_fill(~mask, float("-inf"))


def _ssd(x, A, B, C, chunk: int):
    """Mamba-2 SSD scan, chunk-parallel, float32 throughout for stable exp/cumsum.
    x: (b, l, h, p) already multiplied by dt; A: (b, l, h) = A_head * dt (negative);
    B, C: (b, l, n) shared across heads (single group). Returns (b, l, h, p).
    Autocast is disabled inside: the einsums must not silently drop to bf16 —
    state error would compound across the inter-chunk recurrence."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        return _ssd_f32(x, A, B, C, chunk)


def _ssd_f32(x, A, B, C, chunk: int):
    b, l, h, p = x.shape
    n = B.size(-1)
    c = l // chunk
    x = x.view(b, c, chunk, h, p)
    B = B.view(b, c, chunk, n)
    C = C.view(b, c, chunk, n)
    A = A.view(b, c, chunk, h).permute(0, 3, 1, 2)          # (b, h, c, l)
    A_cs = A.cumsum(-1)
    # intra-chunk (diagonal blocks): two-step einsum keeps intermediates small
    CB = torch.einsum("bcln,bcsn->bcls", C, B)              # (b, c, l, s)
    M = torch.exp(_segsum(A)) * CB.unsqueeze(1)             # (b, h, c, l, s)
    Y = torch.einsum("bhcls,bcshp->bclhp", M, x)
    # per-chunk end states
    decay_states = torch.exp(A_cs[..., -1:] - A_cs)         # (b, h, c, l)
    states = torch.einsum("bcln,bhcl,bclhp->bchpn", B, decay_states, x)
    # inter-chunk recurrence
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)  # (b, c+1, h, p, n)
    decay_chunk = torch.exp(_segsum(F.pad(A_cs[..., -1], (1, 0))))        # (b, h, c+1, c+1)
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)[:, :-1]
    # state -> output
    Y = Y + torch.einsum("bcln,bchpn,bhcl->bclhp", C, states, torch.exp(A_cs))
    return Y.reshape(b, l, h, p)


class MambaBlock(nn.Module):
    """Minimal Mamba-2 mixer: gated selective state space with scalar-per-head A and
    the SSD chunk-parallel scan (no fused kernels). Conv1d/A_log/D/dt_bias/norms are
    non-2-D -> AdamW group; in_proj/out_proj are 2-D block weights -> Muon-class."""

    def __init__(self, dim: int, d_state: int = 64, expand: int = 2, headdim: int = 64,
                 d_conv: int = 4, chunk: int = 64):
        super().__init__()
        self.d_inner = expand * dim
        self.nheads = self.d_inner // headdim
        self.headdim = headdim
        self.d_state = d_state
        self.d_conv = d_conv
        self.chunk = chunk
        self.norm = RMSNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * self.d_inner + 2 * d_state + self.nheads, bias=False)
        conv_ch = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(conv_ch, conv_ch, d_conv, groups=conv_ch, bias=True)
        dt = torch.exp(torch.empty(self.nheads).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(1.0, 16.0)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x):
        Bb, T, _ = x.shape
        z, xBC, dt = self.in_proj(self.norm(x)).split(
            [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1)
        xBC = self.conv1d(F.pad(xBC.transpose(1, 2), (self.d_conv - 1, 0))).transpose(1, 2)
        xBC = F.silu(xBC)
        xs, Bs, Cs = xBC.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(dt.float() + self.dt_bias.float())               # (b, l, h)
        A = -torch.exp(self.A_log.float())                               # (h,)
        xh = xs.view(Bb, T, self.nheads, self.headdim).float()
        y = _ssd(xh * dt.unsqueeze(-1), A * dt, Bs.float(), Cs.float(), self.chunk)
        y = y + self.D.float()[None, None, :, None] * xh                 # skip connection
        y = y.reshape(Bb, T, self.d_inner).to(x.dtype)
        y = self.out_norm(y * F.silu(z))
        return x + self.out_proj(y)


class Mamba(nn.Module):
    """Mamba-2-style SSM LM (untied embeddings, RMSNorm final). The architecture-
    generalization cell: no attention anywhere, same optimizer parameter split."""

    def __init__(self, vocab_size: int, block_size: int, n_layer: int, dim: int):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([MambaBlock(dim) for _ in range(n_layer)])
        self.ln_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def hidden_matrix_names(self):
        return [n for n, p in self.named_parameters() if p.ndim == 2 and n.startswith("blocks.")]


class ViT(nn.Module):
    """Compact ViT for CIFAR: same Block as GPT (non-causal), so the Muon parameter
    treatment is identical — the vision test isolates modality, not parameterization."""

    def __init__(self, num_classes: int = 10, img: int = 32, patch: int = 4,
                 n_layer: int = 6, n_head: int = 4, dim: int = 256):
        super().__init__()
        self.patch = patch
        self.n_tokens = (img // patch) ** 2
        self.embed = nn.Linear(3 * patch * patch, dim, bias=False)
        self.pos_emb = nn.Embedding(self.n_tokens, dim)
        self.blocks = nn.ModuleList([Block(dim, n_head, causal=False) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(dim, bias=False)
        self.head = nn.Linear(dim, num_classes, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 / math.sqrt(2))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, targets=None):
        B = x.size(0)
        p = self.patch
        g = x.size(-1) // p
        x = x.view(B, 3, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(B, self.n_tokens, -1)
        h = self.embed(x) + self.pos_emb(torch.arange(self.n_tokens, device=x.device))
        for b in self.blocks:
            h = b(h)
        logits = self.head(self.ln_f(h).mean(dim=1))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def hidden_matrix_names(self):
        return [n for n, p in self.named_parameters() if p.ndim == 2 and n.startswith("blocks.")]
