import torch
from torch import Tensor

# Оси: B — batch, T — длина последовательности, d — размерность q/k (чётная).


def rope_angles(seq_len: int, d: int, base: float = 10000.0) -> Tensor:
    """
    Углы поворота для каждой позиции и каждой пары координат.

    seq_len: T
    d: размерность q/k, чётная
    Возвращает: (T, d/2), где angles[m, i] = m * theta_i,  theta_i = base^(-2i/d)
    """
    assert d % 2 == 0
    i = torch.arange(d // 2, dtype=torch.float32)    # (d/2,)  номера пар: 0, 1, ..., d/2-1
    theta = base ** (-2 * i / d)                      # (d/2,)  частоты: theta_0 = 1 (быстрая), дальше всё медленнее
    m = torch.arange(seq_len, dtype=torch.float32)    # (T,)    позиции: 0, 1, ..., T-1
    return torch.outer(m, theta)                      # (T, d/2) внешнее произведение: angles[m, i] = m * theta[i]


def sinusoidal_pe(seq_len: int, d: int, base: float = 10000.0) -> Tensor:
    """
    Классическое синусоидальное PE из «Attention Is All You Need» (аддитивное — его ПРИБАВЛЯЮТ к эмбеддингу).

    seq_len: T
    d: размерность, чётная
    Возвращает: (T, d), где
        pe[m, 2i]   = sin(m * theta_i)
        pe[m, 2i+1] = cos(m * theta_i)
    """
    assert d % 2 == 0
    angles = rope_angles(seq_len, d, base)

    pe1 = torch.sin(angles)
    pe2 = torch.cos(angles)

    return torch.stack([pe1, pe2], dim=-1).flatten(-2)


def apply_rope(x: Tensor, angles: Tensor) -> Tensor:
    """
    Поворачивает каждую пару координат (x[..., 0], x[..., 1]), (x[..., 2], x[..., 3]), ...
    токена на позиции m на угол angles[m, i].

    x:      (B, T, d) — q или k
    angles: (T, d/2)  — из rope_angles
    Возвращает: (B, T, d) — та же форма; пары стоят на тех же местах (0,1), (2,3), ...
    """
    a, b = x[..., 0::2], x[..., 1::2]
    cs, sn = torch.cos(angles), torch.sin(angles)
    a_new, b_new = a*cs-b*sn, a*sn+b*cs
    return torch.stack([a_new, b_new], dim=-1).flatten(-2)
