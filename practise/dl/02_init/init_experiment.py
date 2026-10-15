import math

import torch
import torch.nn as nn


def init_std(fan_in, scheme):
    """
    Стандартное отклонение весов для нормальной инициализации N(0, std^2).

    fan_in — сколько входов у одного нейрона слоя (у nn.Linear это in_features).
    Например, у слоя nn.Linear(256, 256) каждый нейрон получает 256 входов, fan_in = 256.

    Слои в эксперименте квадратные (in = out), поэтому у Xavier формула
    2 / (fan_in + fan_out) упрощается до 1 / fan_in.
    """
    if scheme == "xavier":
        return math.sqrt(1 / fan_in)
    if scheme == "he":
        return math.sqrt(2 / fan_in)
    raise ValueError(f"unknown scheme: {scheme}")


def build_net(depth, width, scheme):
    """Список из depth квадратных линейных слоёв без bias. Этот код уже готов."""
    layers = []
    for _ in range(depth):
        layer = nn.Linear(width, width, bias=False)
        # Разобранный пример API: nn.init.normal_ заполняет тензор НА МЕСТЕ
        # числами из N(mean, std^2). Подчёркивание в конце имени = in-place.
        nn.init.normal_(layer.weight, mean=0.0, std=init_std(width, scheme))
        layers.append(layer)
    return layers


@torch.no_grad()  # градиенты не нужны: мы только смотрим на прямой проход
def activation_stds(depth, width, scheme, act, n_samples=512, seed=0):
    """
    Прогоняет случайный батч через сеть и возвращает список std активаций
    после каждого слоя (длина списка = depth).

    act: "relu" или "tanh".
    """
    torch.manual_seed(seed)
    layers = build_net(depth, width, scheme)
    act_fn = torch.relu if act == "relu" else torch.tanh

    h = torch.randn(n_samples, width)  # вход: 512 примеров, std = 1
    stds = []
    for layer in layers:
        h = act_fn(layer(h))
        stds.append(h.std().item())
    return stds
