import torch
from torch import nn


def make_model() -> nn.Module:
    # Готово: маленький MLP для MNIST, 784 -> 128 -> 10
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))


def snapshot(model: nn.Module) -> list[torch.Tensor]:
    # Готово: копия всех параметров модели
    return [p.detach().clone() for p in model.parameters()]


def steps_per_epoch(n_samples: int, batch_size: int, drop_last: bool = False) -> int:
    if not drop_last and n_samples % batch_size != 0:
        return n_samples // batch_size + 1
    else:
        return n_samples // batch_size


def update_norm(before: list[torch.Tensor], after: list[torch.Tensor]) -> float:
    l2_sum = 0
    for a, b in zip(after, before):
        l2_sum += ((a - b)**2).sum().item()

    return l2_sum ** 0.5


def train_one_epoch(model, loader, optimizer, loss_fn, device, scheduler=None):
    # Возвращает (losses, update_norms) — два списка float, по одному значению на шаг
    model.train()
    losses, update_norms = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        before = snapshot(model)
        
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        after = snapshot(model)
        update_norms.append(update_norm(before, after))

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

    return losses, update_norms


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total_correct = 0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device) # torch.Tensor: [targets] | targets = batch_size

        logits = model(x)
        preds = logits.argmax(dim=1)

        bsize = len(y)
        total_n += bsize

        total_correct += (preds == y).sum().item()

    return total_correct / total_n
