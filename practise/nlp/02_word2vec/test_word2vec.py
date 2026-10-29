import math
from collections import Counter

import pytest
import torch

from word2vec import (
    SGNS,
    build_vocab,
    make_toy_corpus,
    most_similar,
    negative_distribution,
    sample_negatives,
    skipgram_pairs,
    train,
)

SENTS = [
    ["кот", "ест", "рыбу"],
    ["кот", "спит"],
    ["пёс", "ест", "мясо", "и", "кот", "ест"],
]


# ---------- build_vocab ----------

def test_vocab_ids_consistent():
    """word2id и id2word — взаимно обратные, id идут 0..V-1, counts длины V."""
    word2id, id2word, counts = build_vocab(SENTS)
    V = len({w for s in SENTS for w in s})
    assert len(id2word) == len(word2id) == V
    assert sorted(word2id.values()) == list(range(V))
    for i, w in enumerate(id2word):
        assert word2id[w] == i
    assert counts.shape == (V,)


def test_vocab_counts_match_manual():
    """counts[word2id[w]] == сколько раз w встретилось во всём корпусе; тип — целочисленный."""
    word2id, id2word, counts = build_vocab(SENTS)
    manual = Counter(w for s in SENTS for w in s)
    assert counts.dtype == torch.long
    for w, c in manual.items():
        assert counts[word2id[w]].item() == c


def test_vocab_min_count_drops_rare():
    """min_count=2: остаются только «кот» (3) и «ест» (3), id всё равно 0..V-1."""
    word2id, id2word, counts = build_vocab(SENTS, min_count=2)
    assert set(word2id) == {"кот", "ест"}
    assert sorted(word2id.values()) == [0, 1]
    assert len(id2word) == 2
    assert counts[word2id["кот"]].item() == 3 and counts[word2id["ест"]].item() == 3


# ---------- skipgram_pairs ----------

def test_pairs_window1_exact_order():
    """[0..4], window=1: 8 пар; по t слева направо, внутри t — сначала левый сосед, потом правый."""
    assert skipgram_pairs([0, 1, 2, 3, 4], window=1) == [
        (0, 1),
        (1, 0), (1, 2),
        (2, 1), (2, 3),
        (3, 2), (3, 4),
        (4, 3),
    ]


def test_pairs_window2_count_and_edges():
    """[0..4], window=2: 2+3+4+3+2 = 14 пар; window больше предложения — края не выходят за его пределы."""
    pairs = skipgram_pairs([0, 1, 2, 3, 4], window=2)
    assert len(pairs) == 14
    assert pairs[:2] == [(0, 1), (0, 2)]
    assert pairs[5:9] == [(2, 0), (2, 1), (2, 3), (2, 4)]
    # window=5 на 3 словах: каждый с каждым, без самого себя и без «чужих» id
    assert skipgram_pairs([10, 20, 30], window=5) == [
        (10, 20), (10, 30),
        (20, 10), (20, 30),
        (30, 10), (30, 20),
    ]


def test_pairs_single_word_empty():
    """Предложение из одного слова — контекста нет, пар нет."""
    assert skipgram_pairs([7], window=2) == []


# ---------- negative_distribution ----------

def test_neg_dist_formula_and_sum():
    """Совпадает с counts**0.75 / sum, сумма = 1, тип — float."""
    counts = torch.tensor([100, 10, 1, 50])
    p = negative_distribution(counts)
    expected = counts.float() ** 0.75 / (counts.float() ** 0.75).sum()
    assert p.dtype.is_floating_point
    assert p.shape == (4,)
    assert torch.isclose(p.sum(), torch.tensor(1.0))
    assert torch.allclose(p, expected)


def test_neg_dist_flatter_than_counts():
    """Степень 0.75 сглаживает: p(частое)/p(редкое) меньше, чем counts(частое)/counts(редкое)."""
    counts = torch.tensor([1000, 10])
    p = negative_distribution(counts)
    assert (p[0] / p[1]).item() < 1000 / 10
    assert (p[0] / p[1]).item() == pytest.approx(100 ** 0.75, rel=1e-4)


def test_neg_dist_power1_is_frequencies():
    """power=1 — обычные частоты counts / sum."""
    counts = torch.tensor([3, 1, 4, 2])
    p = negative_distribution(counts, power=1.0)
    assert torch.allclose(p, torch.tensor([0.3, 0.1, 0.4, 0.2]))


# ---------- SGNS ----------

def test_sgns_init():
    """center ~ U(-0.5/d, 0.5/d) (не нули), context — нули; оба nn.Embedding (V, d)."""
    torch.manual_seed(0)
    V, d = 50, 8
    model = SGNS(V, d)
    assert isinstance(model.center, torch.nn.Embedding)
    assert isinstance(model.context, torch.nn.Embedding)
    assert model.center.weight.shape == (V, d) and model.context.weight.shape == (V, d)
    w = model.center.weight
    assert w.abs().max().item() <= 0.5 / d
    assert w.abs().max().item() > 0.25 / d  # на 400 числах максимум точно в верхней половине интервала
    assert torch.all(model.context.weight == 0)


def test_sgns_start_loss_is_k_plus_1_ln2():
    """На старте u = 0 → все u·v = 0 → σ(0) = 0.5 → loss = (1 + K)·ln 2. Форма — скаляр ()."""
    torch.manual_seed(0)
    V, d, B, K = 20, 8, 6, 5
    model = SGNS(V, d)
    center = torch.randint(0, V, (B,))
    pos = torch.randint(0, V, (B,))
    neg = torch.randint(0, V, (B, K))
    loss = model(center, pos, neg)
    assert loss.shape == ()
    assert loss.item() == pytest.approx((1 + K) * math.log(2), rel=1e-6)


def test_sgns_loss_matches_formula_on_manual_weights():
    """С заданными вручную векторами loss = среднее по batch от
    -log σ(u_pos·v_c) - Σ_i log σ(-u_neg_i·v_c)."""
    model = SGNS(4, 2)
    v = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 0.5]])
    u = torch.tensor([[0.5, 0.5], [1.0, -1.0], [-0.5, 2.0], [0.0, 1.0]])
    with torch.no_grad():
        model.center.weight.copy_(v)
        model.context.weight.copy_(u)
    center = torch.tensor([0, 1])
    pos = torch.tensor([2, 3])
    neg = torch.tensor([[1, 3], [0, 2]])

    def log_sigmoid(x: float) -> float:
        return -math.log1p(math.exp(-x))

    def dot(a, b) -> float:
        return sum(x * y for x, y in zip(a.tolist(), b.tolist()))

    per_pair = []
    for b in range(2):
        vc = v[center[b]]
        j = -log_sigmoid(dot(u[pos[b]], vc))
        for n in neg[b]:
            j -= log_sigmoid(-dot(u[n], vc))
        per_pair.append(j)
    expected = sum(per_pair) / len(per_pair)

    assert model(center, pos, neg).item() == pytest.approx(expected, rel=1e-5)


def test_sgns_grad_only_on_batch_center_rows():
    """После backward у center.weight ненулевой градиент только в строках слов из center."""
    torch.manual_seed(0)
    V, d = 10, 4
    model = SGNS(V, d)
    with torch.no_grad():
        model.context.weight.normal_()  # иначе u = 0 и градиент по v тоже нулевой
    center = torch.tensor([1, 3, 3])
    pos = torch.tensor([2, 5, 7])
    neg = torch.tensor([[0, 4], [6, 8], [9, 2]])
    model(center, pos, neg).backward()
    g = model.center.weight.grad
    nonzero_rows = {i for i in range(V) if g[i].abs().sum().item() > 0}
    assert nonzero_rows == {1, 3}


# ---------- sample_negatives (готовая функция, проверка формы) ----------

def test_sample_negatives_shape_and_range():
    """Форма (B, K), id в 0..V-1, при одном seed — одинаковый результат."""
    dist = torch.tensor([0.1, 0.2, 0.3, 0.4])
    a = sample_negatives(dist, 7, 3, torch.Generator().manual_seed(0))
    b = sample_negatives(dist, 7, 3, torch.Generator().manual_seed(0))
    assert a.shape == (7, 3)
    assert a.min() >= 0 and a.max() < 4
    assert torch.equal(a, b)


# ---------- train ----------

@pytest.fixture(scope="module")
def toy_setup():
    sents = make_toy_corpus(seed=0)
    word2id, id2word, counts = build_vocab(sents)
    pairs = [p for s in sents for p in skipgram_pairs([word2id[w] for w in s], window=2)]
    return word2id, id2word, torch.tensor(pairs), negative_distribution(counts)


def test_train_returns_loss_per_epoch(toy_setup):
    """Список float длины epochs."""
    word2id, _, pairs, dist = toy_setup
    torch.manual_seed(0)
    model = SGNS(len(word2id), 8)
    losses = train(model, pairs[:2000], dist, epochs=3, batch_size=256, k=3, lr=0.01)
    assert len(losses) == 3
    assert all(isinstance(x, float) for x in losses)


def test_train_loss_decreases_on_toy_corpus(toy_setup):
    """На make_toy_corpus(seed=0) loss последней эпохи заметно меньше первой."""
    word2id, _, pairs, dist = toy_setup
    torch.manual_seed(0)
    model = SGNS(len(word2id), 8)
    losses = train(model, pairs, dist, epochs=4, batch_size=512, k=5, lr=0.01)
    assert losses[-1] < 0.9 * losses[0]


# ---------- most_similar ----------

@pytest.fixture
def manual_model():
    """a=(1,0); b=(2,0.2) почти тот же угол; c=(10,10) под 45°, но с большой нормой; d=(-1,0) напротив."""
    id2word = ["a", "b", "c", "d"]
    word2id = {w: i for i, w in enumerate(id2word)}
    model = SGNS(4, 2)
    with torch.no_grad():
        model.center.weight.copy_(torch.tensor([[1.0, 0.0], [2.0, 0.2], [10.0, 10.0], [-1.0, 0.0]]))
    return model, word2id, id2word


def test_most_similar_excludes_self_sorted(manual_model):
    """Самого слова нет, длина k, косинусы по убыванию."""
    model, word2id, id2word = manual_model
    res = most_similar(model, "a", word2id, id2word, k=3)
    assert len(res) == 3
    assert "a" not in [w for w, _ in res]
    sims = [s for _, s in res]
    assert sims == sorted(sims, reverse=True)


def test_most_similar_known_answer_cosine_not_dot(manual_model):
    """По косинусу b ближе c (по скалярному произведению было бы наоборот); значения — косинусы."""
    model, word2id, id2word = manual_model
    res = most_similar(model, "a", word2id, id2word, k=3)
    assert [w for w, _ in res] == ["b", "c", "d"]
    assert res[0][1] == pytest.approx(2 / math.sqrt(4.04), rel=1e-5)
    assert res[1][1] == pytest.approx(1 / math.sqrt(2), rel=1e-5)
    assert res[2][1] == pytest.approx(-1.0, rel=1e-5)
    assert [w for w, _ in most_similar(model, "a", word2id, id2word, k=1)] == ["b"]
