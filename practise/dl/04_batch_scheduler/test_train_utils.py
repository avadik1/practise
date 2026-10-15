import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_utils import evaluate, make_model, snapshot, steps_per_epoch, train_one_epoch, update_norm

DEVICE = torch.device("cpu")


def fake_loader(n=100, batch_size=16):
    torch.manual_seed(0)
    x = torch.randn(n, 1, 28, 28)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


@pytest.mark.parametrize("n,bs,drop,expected", [
    (60000, 32, False, 1875),
    (60000, 128, False, 469),
    (60000, 128, True, 468),
    (100, 16, False, 7),
    (100, 16, True, 6),
])
def test_steps_per_epoch(n, bs, drop, expected):
    assert steps_per_epoch(n, bs, drop) == expected


def test_update_norm():
    before = [torch.zeros(2), torch.zeros(1)]
    after = [torch.tensor([3.0, 0.0]), torch.tensor([4.0])]
    assert update_norm(before, after) == pytest.approx(5.0)
    assert update_norm(before, before) == 0.0


def test_train_one_epoch_outputs():
    torch.manual_seed(0)
    model = make_model()
    loader = fake_loader(100, 16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = snapshot(model)
    losses, norms = train_one_epoch(model, loader, opt, nn.CrossEntropyLoss(), DEVICE)
    assert len(losses) == len(norms) == steps_per_epoch(100, 16)
    assert all(isinstance(v, float) for v in losses + norms)
    assert all(v > 0 for v in norms)
    assert update_norm(before, snapshot(model)) > 0


def test_scheduler_steps_every_batch():
    torch.manual_seed(0)
    model = make_model()
    loader = fake_loader(100, 16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(loader))
    train_one_epoch(model, loader, opt, nn.CrossEntropyLoss(), DEVICE, scheduler=sched)
    assert sched.last_epoch == len(loader)
    assert opt.param_groups[0]["lr"] < 1e-6


class CheatModel(nn.Module):
    # Метка закодирована в x[:, 0, 0, :10] как one-hot -> идеальная модель
    def forward(self, x):
        return x[:, 0, 0, :10]


def test_evaluate():
    acc = evaluate(make_model(), fake_loader(), DEVICE)
    assert 0.0 <= acc <= 1.0

    y = torch.arange(10).repeat(5)
    x = torch.zeros(50, 1, 28, 28)
    x[torch.arange(50), 0, 0, y] = 1.0
    perfect = DataLoader(TensorDataset(x, y), batch_size=8)
    assert evaluate(CheatModel(), perfect, DEVICE) == pytest.approx(1.0)
