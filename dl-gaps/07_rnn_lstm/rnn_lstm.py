import torch
from torch import Tensor

# Оси: B — batch, T — длина последовательности, D — размер входа x_t, H — размер h_t (и c_t).


def rnn_cell(x_t: Tensor, h_prev: Tensor, W_x: Tensor, W_h: Tensor, b: Tensor) -> Tensor:
    """
    Один шаг обычной RNN:  h_t = tanh(x_t W_x + h_prev W_h + b)

    x_t:    (B, D)
    h_prev: (B, H)
    W_x:    (D, H)
    W_h:    (H, H)
    b:      (H,)
    Возвращает h_t: (B, H)
    """
    return torch.tanh(x_t @ W_x + h_prev @ W_h + b)


def lstm_cell(
    x_t: Tensor, h_prev: Tensor, c_prev: Tensor, W: Tensor, b: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Один шаг LSTM. Все 4 блока считаются ОДНИМ матричным умножением от [x_t, h_prev].

    x_t:    (B, D)
    h_prev: (B, H)
    c_prev: (B, H)
    W:      (D + H, 4H)  — строки: сначала D строк под x_t, потом H строк под h_prev
    b:      (4H,)
    Порядок блоков по последней оси (как в torch.nn.LSTMCell): i, f, g, o
        i — input gate (сигмоида), f — forget gate (сигмоида),
        g — кандидат c~_t (tanh),  o — output gate (сигмоида)

    Возвращает (h_t, c_t): обе (B, H)
    """
    z = torch.cat([x_t, h_prev], dim=-1)
    z = z @ W + b

    i, f, g, o = torch.chunk(z, chunks=4, dim=-1)
    i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
    g = torch.tanh(g)

    c_t = f * c_prev + i * g
    h_t = o * torch.tanh(c_t)

    return h_t, c_t


def rnn_forward(xs: Tensor, h0: Tensor, W_x: Tensor, W_h: Tensor, b: Tensor) -> list[Tensor]:
    """
    Прогон RNN по всей последовательности.

    xs: (B, T, D)
    h0: (B, H)
    Возвращает список длины T: [h_1, ..., h_T], каждый (B, H).
    Важно: в список кладутся ИМЕННО те тензоры, что идут на следующий шаг (без .clone()/.detach()),
    иначе по ним нельзя будет взять градиент.
    """
    hs = list()
    h = h0

    for t in range(xs.size(1)):
        x_t = xs[:, t]
        h = rnn_cell(x_t, h, W_x, W_h, b)
        hs.append(h)

    return hs


def lstm_forward(
    xs: Tensor, h0: Tensor, c0: Tensor, W: Tensor, b: Tensor
) -> tuple[list[Tensor], list[Tensor]]:
    """
    Прогон LSTM по всей последовательности.

    xs: (B, T, D),  h0, c0: (B, H)
    Возвращает (hs, cs): два списка длины T, каждый элемент (B, H).
    То же правило: в списках — ровно те тензоры, что идут на следующий шаг.
    """
    hs, cs = list(), list()
    h, c = h0, c0

    for t in range(xs.size(1)):
        x_t = xs[:, t]
        h, c = lstm_cell(x_t, h, c, W, b)
        hs.append(h)
        cs.append(c)

    return hs, cs


def state_grad_norms(states: list[Tensor], loss: Tensor) -> Tensor:
    """
    Для каждого промежуточного состояния s_k считает норму dL/ds_k.

    states: список длины T, каждый (B, H) — промежуточные тензоры графа (не листья!)
    loss:   скаляр
    Возвращает: (T,) — states[k] -> ||dL/d states[k]||
    """
    grads = torch.autograd.grad(loss, states, retain_graph=True, allow_unused=True)
    norms = [
        torch.zeros(()) if g is None else g.norm()   # норма по всем элементам (B, H)
        for g in grads
    ]
    return torch.stack(norms)                          # (T,)


def init_rnn(D: int, H: int, gain: float = 1.0, seed: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    """W_x (D, H), W_h (H, H), b (H,). gain масштабирует W_h."""
    g = torch.Generator().manual_seed(seed)
    W_x = torch.randn(D, H, generator=g) / D**0.5
    W_h = gain * torch.randn(H, H, generator=g) / H**0.5
    b = torch.zeros(H)
    return W_x, W_h, b


def init_lstm(D: int, H: int, forget_bias: float = 0.0, seed: int = 0) -> tuple[Tensor, Tensor]:
    """W (D+H, 4H), b (4H,). forget_bias кладётся в блок f (второй из i, f, g, o)."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(D + H, 4 * H, generator=g) / (D + H) ** 0.5
    b = torch.zeros(4 * H)
    b[H : 2 * H] = forget_bias
    return W, b
