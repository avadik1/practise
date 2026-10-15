import copy

# Конфиг: словарь, внутри которого есть вложенный словарь "model".
# base = {"model": {"lr": 1e-3, "optimizer": "adam"}, "epochs": 10}


def make_configs(base: dict, lrs: list[float]) -> list[dict]:
    """
    Строит по одному конфигу на каждый lr из lrs.

    base: базовый конфиг; НЕ должен измениться после вызова
    lrs:  список learning rate, например [1e-2, 1e-3, 1e-4]
    Возвращает: список длины len(lrs); в i-м конфиге cfg["model"]["lr"] == lrs[i],
                всё остальное — как в base. Конфиги независимы друг от друга и от base.
    """
    configs = list()

    for lr in lrs:
        copy_base = copy.deepcopy(base)
        copy_base["model"]["lr"] = lr
        configs.append(copy_base)

    return configs

