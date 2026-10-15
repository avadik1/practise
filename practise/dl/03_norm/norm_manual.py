import torch


def my_batchnorm(x: torch.Tensor, eps=1e-5) -> torch.Tensor:
    """
    BatchNorm в режиме обучения, без обучаемых gamma и beta.

    x: тензор (batch, features).
    Каждый ПРИЗНАК нормализуется по всем ПРИМЕРАМ батча:
    из столбца вычитается его среднее и делится на его std.
    """
    mean = torch.mean(x, dim=0, keepdim=True)
    var = torch.var(x, dim=0, unbiased=False, keepdim=True)
    return (x-mean) / (torch.sqrt(var + eps))


def my_layernorm(x: torch.Tensor, eps=1e-5) -> torch.Tensor:
    """
    LayerNorm без обучаемых gamma и beta.

    x: тензор (batch, features).
    Каждый ПРИМЕР нормализуется по всем своим ПРИЗНАКАМ:
    из строки вычитается её среднее и делится на её std.
    """
    mean = torch.mean(x, dim=1, keepdim=True)
    var = torch.var(x, dim=1, unbiased=False, keepdim=True)
    return (x-mean) / (torch.sqrt(var + eps))


def normalized_example(norm_fn, x: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    """
    x: один пример, тензор (features,).
    others: другие примеры, тензор (k, features).

    Собирает батч из x и others, нормализует его функцией norm_fn
    и возвращает только то, во что превратился x.
    """
    x = torch.unsqueeze(x, 0)
    batch = torch.cat((x, others), dim=0)
    norm_batch = norm_fn(batch)

    return norm_batch[0]