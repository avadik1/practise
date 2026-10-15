import torch
import torch.nn as nn

from norm_manual import my_batchnorm, my_layernorm, normalized_example

torch.manual_seed(0)
X = torch.randn(8, 16) * 3 + 5  # 8 примеров, 16 признаков, не нормализованы


def test_bn_matches_torch():
    ref = nn.BatchNorm1d(16, affine=False, track_running_stats=False)(X)
    assert torch.allclose(my_batchnorm(X), ref, atol=1e-5)


def test_ln_matches_torch():
    ref = nn.LayerNorm(16, elementwise_affine=False)(X)
    assert torch.allclose(my_layernorm(X), ref, atol=1e-5)


def test_shapes_preserved():
    assert my_batchnorm(X).shape == X.shape
    assert my_layernorm(X).shape == X.shape


def test_normalized_example_shape():
    out = normalized_example(my_layernorm, X[0], X[1:4])
    assert out.shape == (16,)


def test_bn_output_depends_on_batch():
    a = normalized_example(my_batchnorm, X[0], X[1:4])
    b = normalized_example(my_batchnorm, X[0], X[4:8])
    assert not torch.allclose(a, b, atol=1e-3)


def test_ln_output_independent_of_batch():
    a = normalized_example(my_layernorm, X[0], X[1:4])
    b = normalized_example(my_layernorm, X[0], X[4:8])
    assert torch.allclose(a, b, atol=1e-6)
