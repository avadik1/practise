import torch

from rnn_lstm import (
    init_lstm, init_rnn, lstm_cell, lstm_forward, rnn_cell, rnn_forward, state_grad_norms,
)

B, T, D, H = 3, 6, 4, 5


def test_rnn_cell_shape():
    W_x, W_h, b = init_rnn(D, H)
    h = rnn_cell(torch.randn(B, D), torch.randn(B, H), W_x, W_h, b)
    assert h.shape == (B, H)


def test_rnn_cell_matches_torch():
    torch.manual_seed(0)
    cell = torch.nn.RNNCell(D, H)                     # tanh по умолчанию
    x, h = torch.randn(B, D), torch.randn(B, H)
    ours = rnn_cell(x, h, cell.weight_ih.T, cell.weight_hh.T, cell.bias_ih + cell.bias_hh)
    assert torch.allclose(ours, cell(x, h), atol=1e-6)


def test_lstm_cell_shapes():
    W, b = init_lstm(D, H)
    h, c = lstm_cell(torch.randn(B, D), torch.randn(B, H), torch.randn(B, H), W, b)
    assert h.shape == (B, H) and c.shape == (B, H)


def test_lstm_cell_matches_torch():
    torch.manual_seed(0)
    cell = torch.nn.LSTMCell(D, H)
    x, h, c = torch.randn(B, D), torch.randn(B, H), torch.randn(B, H)
    W = torch.cat([cell.weight_ih.T, cell.weight_hh.T], dim=0)   # (D+H, 4H)
    b = cell.bias_ih + cell.bias_hh
    h_ours, c_ours = lstm_cell(x, h, c, W, b)
    h_ref, c_ref = cell(x, (h, c))
    assert torch.allclose(h_ours, h_ref, atol=1e-6)
    assert torch.allclose(c_ours, c_ref, atol=1e-6)


def test_lstm_keeps_memory_when_f1_i0():
    # f ≈ 1 (огромный bias), i ≈ 0 (огромный минус) -> c_t == c_prev
    W, b = init_lstm(D, H)
    W = torch.zeros_like(W)
    b = torch.zeros(4 * H)
    b[:H] = -50.0        # i
    b[H:2 * H] = 50.0    # f
    c_prev = torch.randn(B, H)
    _, c = lstm_cell(torch.randn(B, D), torch.randn(B, H), c_prev, W, b)
    assert torch.allclose(c, c_prev, atol=1e-6)


def test_rnn_forward_matches_torch():
    torch.manual_seed(0)
    rnn = torch.nn.RNN(D, H, batch_first=True)
    xs, h0 = torch.randn(B, T, D), torch.zeros(B, H)
    hs = rnn_forward(xs, h0, rnn.weight_ih_l0.T, rnn.weight_hh_l0.T, rnn.bias_ih_l0 + rnn.bias_hh_l0)
    out, _ = rnn(xs, h0.unsqueeze(0))
    assert len(hs) == T
    assert torch.allclose(torch.stack(hs, dim=1), out, atol=1e-5)


def test_lstm_forward_lists():
    W, b = init_lstm(D, H)
    hs, cs = lstm_forward(torch.randn(B, T, D), torch.zeros(B, H), torch.zeros(B, H), W, b)
    assert len(hs) == T and len(cs) == T
    assert all(h.shape == (B, H) for h in hs) and all(c.shape == (B, H) for c in cs)


def test_states_are_in_graph():
    # если в forward есть clone/detach, градиент по ранним состояниям будет None -> 0
    W_x, W_h, b = init_rnn(D, H)
    W_x.requires_grad_(True)
    hs = rnn_forward(torch.randn(B, T, D), torch.zeros(B, H), W_x, W_h, b)
    norms = state_grad_norms(hs, hs[-1].sum())
    assert norms.shape == (T,)
    assert (norms > 0).all()
