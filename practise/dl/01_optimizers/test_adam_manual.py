import torch
from adam_manual import adam_step

torch.manual_seed(0)
P0 = torch.randn(5)
GRADS = [torch.randn(5) for _ in range(20)]


def run_torch(opt_cls, p0, grads, **kw):
    """Прогоняет встроенный оптимизатор torch на заранее заданных градиентах."""
    p = p0.clone().requires_grad_(True)
    opt = opt_cls([p], **kw)
    for g in grads:
        opt.zero_grad()
        p.grad = g.clone()
        opt.step()
    return p.detach()


def run_manual(p0, grads, **kw):
    """Прогоняет твой adam_step на тех же градиентах."""
    p = p0.clone()
    m = torch.zeros_like(p)
    v = torch.zeros_like(p)
    for t, g in enumerate(grads, start=1):
        p, m, v = adam_step(p, g, m, v, t, **kw)
    return p


def test_adam_matches_torch():
    ref = run_torch(torch.optim.Adam, P0, GRADS, lr=1e-2)
    out = run_manual(P0, GRADS, lr=1e-2, mode="adam")
    assert torch.allclose(out, ref, atol=1e-6)


def test_adam_l2_matches_torch():
    ref = run_torch(torch.optim.Adam, P0, GRADS, lr=1e-2, weight_decay=0.1)
    out = run_manual(P0, GRADS, lr=1e-2, weight_decay=0.1, mode="adam")
    assert torch.allclose(out, ref, atol=1e-6)


def test_adamw_matches_torch():
    ref = run_torch(torch.optim.AdamW, P0, GRADS, lr=1e-2, weight_decay=0.1)
    out = run_manual(P0, GRADS, lr=1e-2, weight_decay=0.1, mode="adamw")
    assert torch.allclose(out, ref, atol=1e-6)


def test_l2_is_not_adamw():
    a = run_manual(P0, GRADS, lr=1e-2, weight_decay=0.1, mode="adam")
    b = run_manual(P0, GRADS, lr=1e-2, weight_decay=0.1, mode="adamw")
    assert not torch.allclose(a, b, atol=1e-4)
