import math

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from torch import Tensor

# Оси: B — batch, H — головы, T — токены, d_model = H * d_head.
# *batch — любые ведущие оси, # — ось может растянуться (broadcast).


@jaxtyped(typechecker=beartype)
def scaled_dot_product_attention(
    Q: Float[Tensor, "*batch T_q d_k"],
    K: Float[Tensor, "*batch T_k d_k"],
    V: Float[Tensor, "*batch T_k d_v"],
    mask: Bool[Tensor, "*#batch #T_q #T_k"] | None = None,
) -> tuple[Float[Tensor, "*batch T_q d_v"], Float[Tensor, "*batch T_q T_k"]]:
    """mask: True — позицию можно смотреть (как в F.scaled_dot_product_attention)."""
    d_k = K.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = weights @ V
    return out, weights


@jaxtyped(typechecker=beartype)
def causal_mask(T: int, device: torch.device | str | None = None) -> Bool[Tensor, "T T"]:
    """Позиция i видит только позиции j <= i."""
    allowed = torch.ones(T, T, dtype=torch.bool, device=device)
    return torch.tril(allowed)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Float[Tensor, "B T d_model"]) -> Float[Tensor, "B H T d_head"]:
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Float[Tensor, "B H T d_head"]) -> Float[Tensor, "B T d_model"]:
        B, H, T, d_head = x.size()
        # после permute память не непрерывна — без contiguous view упадёт
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, T, H*d_head)

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[Tensor, "B T d_model"],
        mask: Bool[Tensor, "*#batch #T_q #T_k"] | None = None,
    ) -> tuple[Float[Tensor, "B T d_model"], Float[Tensor, "B H T T"]]:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q, K, V = self._split_heads(Q), self._split_heads(K), self._split_heads(V)

        out, weights = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)

        out = self._merge_heads(out)

        return (self.W_o(out), weights)
