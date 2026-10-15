import math

import torch
import torch.nn as nn

from init_experiment import activation_stds, init_std


def test_init_std_values():
    assert math.isclose(init_std(256, "xavier"), math.sqrt(1 / 256))
    assert math.isclose(init_std(256, "he"), math.sqrt(2 / 256))


def test_xavier_matches_torch():
    torch.manual_seed(0)
    w = torch.empty(1024, 1024)
    nn.init.xavier_normal_(w)
    assert abs(w.std().item() - init_std(1024, "xavier")) < 1e-3


def test_he_matches_torch():
    torch.manual_seed(0)
    w = torch.empty(1024, 1024)
    nn.init.kaiming_normal_(w, nonlinearity="relu")
    assert abs(w.std().item() - init_std(1024, "he")) < 1e-3


def test_one_float_per_layer():
    stds = activation_stds(depth=10, width=64, scheme="he", act="relu")
    assert len(stds) == 10
    assert all(isinstance(s, float) for s in stds)


def test_xavier_relu_vanishes():
    stds = activation_stds(depth=30, width=256, scheme="xavier", act="relu")
    assert stds[-1] < 1e-3


def test_he_relu_keeps_scale():
    stds = activation_stds(depth=30, width=256, scheme="he", act="relu")
    assert 0.1 < stds[-1] < 10
