"""Запусти после того, как тесты пройдут. Ничего дописывать не нужно."""
import torch
from adam_manual import adam_step

lr = 1e-2
g = torch.tensor([0.5])  # постоянный градиент

for bc in (True, False):
    p = torch.zeros(1)
    m = torch.zeros(1)
    v = torch.zeros(1)
    print(f"\nbias_correction={bc}")
    for t in range(1, 6):
        p_old = p.clone()
        p, m, v = adam_step(p, g, m, v, t, lr=lr, bias_correction=bc)
        step = (p_old - p).item()
        print(f"  t={t}: шаг = {step:.5f}  ({step / lr:.2f} * lr)")
