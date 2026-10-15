import torch


def adam_step(p, g, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
              weight_decay=0.0, mode="adam", bias_correction=True):
    """
    Один шаг Adam / AdamW для тензора p.

    p, g, m, v — тензоры одной формы (веса, градиент, 1-й и 2-й моменты).
    t — номер шага, начинается с 1.
    mode="adam"  — weight_decay работает как L2 в loss (так делает torch.optim.Adam)
    mode="adamw" — decoupled weight decay (так делает torch.optim.AdamW)

    Возвращает (p_new, m_new, v_new). Входные тензоры на месте не меняем.
    """
    if mode == "adam" and weight_decay != 0:
        g = g + weight_decay*p

    if mode == "adamw" and weight_decay != 0:
        p = p - lr*weight_decay*p

    m = beta1*m + (1-beta1)*g

    v = beta2*v + (1-beta2)*(g**2)

    if bias_correction:
        m_hat = m / (1-beta1**t)
        v_hat = v / (1-beta2**t)
    else:
        m_hat, v_hat = m, v

    p = p - lr*((m_hat / (torch.sqrt(v_hat)+eps)))

    return p, m, v
