import math

import torch

from rope import apply_rope, rope_angles, sinusoidal_pe


def test_angles_shape_and_values():
    a = rope_angles(4, 6)
    assert a.shape == (4, 3)
    assert torch.allclose(a[0], torch.zeros(3))          # позиция 0 — нулевой угол
    assert torch.allclose(a[1, 0], torch.tensor(1.0))    # theta_0 = 1


def test_sinusoidal_pe():
    pe = sinusoidal_pe(4, 6)
    assert pe.shape == (4, 6)
    assert torch.allclose(pe[0, 0::2], torch.zeros(3))   # sin(0) = 0
    assert torch.allclose(pe[0, 1::2], torch.ones(3))    # cos(0) = 1
    assert pe.abs().max() <= 1.0 + 1e-6


def test_shape_preserved():
    x = torch.randn(2, 5, 8)
    assert apply_rope(x, rope_angles(5, 8)).shape == (2, 5, 8)


def test_position_zero_unchanged():
    x = torch.randn(1, 3, 8)
    y = apply_rope(x, rope_angles(3, 8))
    assert torch.allclose(y[:, 0], x[:, 0], atol=1e-6)


def test_norm_preserved():
    x = torch.randn(2, 7, 16)
    y = apply_rope(x, rope_angles(7, 16))
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_2d_manual_rotation():
    # d = 2, позиция 1 -> угол 1 радиан; вектор (1, 0) -> (cos 1, sin 1)
    x = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    y = apply_rope(x, rope_angles(2, 2))
    assert torch.allclose(y[0, 1], torch.tensor([math.cos(1), math.sin(1)]), atol=1e-6)


def test_pairs_stay_in_place():
    # d = 4: пара (0,1) крутится с theta_0 = 1, пара (2,3) с theta_1 = 10000^(-1/2) = 0.01
    x = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]]])
    y = apply_rope(x, rope_angles(2, 4))
    expected = torch.tensor([math.cos(1), math.sin(1), math.cos(0.01), math.sin(0.01)])
    assert torch.allclose(y[0, 1], expected, atol=1e-6)


def test_score_depends_only_on_distance():
    torch.manual_seed(0)
    d, T = 16, 120
    q, k = torch.randn(d), torch.randn(d)
    angles = rope_angles(T, d)
    Q = apply_rope(q.expand(1, T, d).clone(), angles)[0]   # один и тот же q на всех позициях
    K = apply_rope(k.expand(1, T, d).clone(), angles)[0]   # один и тот же k на всех позициях
    assert torch.allclose(Q[2] @ K[5], Q[100] @ K[103], atol=1e-4)
