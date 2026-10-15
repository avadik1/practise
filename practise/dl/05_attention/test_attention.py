import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import scaled_dot_product_attention, causal_mask, MultiHeadAttention


def test_sdpa_matches_torch():
    torch.manual_seed(0)
    Q = torch.randn(2, 5, 8)
    K = torch.randn(2, 7, 8)
    V = torch.randn(2, 7, 4)
    out, w = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (2, 5, 4)
    assert w.shape == (2, 5, 7)
    ref = F.scaled_dot_product_attention(Q, K, V)
    assert torch.allclose(out, ref, atol=1e-5)


def test_weights_sum_to_one():
    torch.manual_seed(0)
    Q, K, V = torch.randn(3, 4, 8), torch.randn(3, 6, 8), torch.randn(3, 6, 8)
    _, w = scaled_dot_product_attention(Q, K, V)
    assert torch.allclose(w.sum(dim=-1), torch.ones(3, 4), atol=1e-5)


def test_causal_mask():
    m = causal_mask(4)
    expected = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ], dtype=torch.bool)
    assert m.dtype == torch.bool
    assert torch.equal(m, expected)


def test_sdpa_with_mask_matches_torch():
    torch.manual_seed(0)
    Q, K, V = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    mask = causal_mask(5)
    out, w = scaled_dot_product_attention(Q, K, V, mask)
    ref = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
    assert torch.allclose(out, ref, atol=1e-5)
    assert torch.all(w[:, ~mask] == 0), "на запрещённых позициях вес должен быть ровно 0"


def test_mha_shapes():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    out, w = mha(torch.randn(3, 10, 32))
    assert out.shape == (3, 10, 32)
    assert w.shape == (3, 4, 10, 10)


def test_mha_matches_torch():
    torch.manual_seed(0)
    d, H = 16, 4
    mha = MultiHeadAttention(d, H)
    ref = nn.MultiheadAttention(d, H, batch_first=True)
    with torch.no_grad():
        Wq, Wk, Wv = ref.in_proj_weight.chunk(3)
        bq, bk, bv = ref.in_proj_bias.chunk(3)
        mha.W_q.weight.copy_(Wq); mha.W_q.bias.copy_(bq)
        mha.W_k.weight.copy_(Wk); mha.W_k.bias.copy_(bk)
        mha.W_v.weight.copy_(Wv); mha.W_v.bias.copy_(bv)
        mha.W_o.weight.copy_(ref.out_proj.weight); mha.W_o.bias.copy_(ref.out_proj.bias)
    x = torch.randn(2, 5, d)
    out, _ = mha(x)
    ref_out, _ = ref(x, x, x, need_weights=False)
    assert torch.allclose(out, ref_out, atol=1e-5)


def test_causal_no_future_leak():
    torch.manual_seed(1)
    mha = MultiHeadAttention(16, 4)
    x = torch.randn(1, 6, 16)
    x2 = x.clone()
    x2[:, 4:] = torch.randn(1, 2, 16)   # меняем только "будущие" токены 4 и 5
    mask = causal_mask(6)
    out1, _ = mha(x, mask)
    out2, _ = mha(x2, mask)
    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-6), "прошлые позиции подсмотрели будущее"
    assert not torch.allclose(out1[:, 4:], out2[:, 4:])
