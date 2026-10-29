"""Word2Vec с нуля: Skip-Gram + negative sampling (SGNS) на PyTorch."""
import random

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from jaxtyping import Float, Int

# Оси: V — размер словаря, d — размерность эмбеддинга, B — batch (число пар), K — негативов на пару,
#      P — число обучающих пар во всём корпусе.

TOPICS: dict[str, list[str]] = {
    "животные": ["кошка", "собака", "лошадь", "корова", "овца", "коза", "кролик", "лиса", "ёж"],
    "еда": ["хлеб", "сыр", "молоко", "суп", "каша", "яблоко", "мясо", "рыба", "икра"],
    "транспорт": ["машина", "автобус", "поезд", "самолёт", "трамвай", "велосипед", "метро", "такси", "паром"],
}
RARE: dict[str, str] = {"животные": "ёж", "еда": "икра", "транспорт": "паром"}  # последнее слово каждой темы
FUNCTION_WORDS: list[str] = ["и", "в", "на", "очень", "где"]


# ---------- готово ----------
def make_toy_corpus(n_sentences: int = 3000, seed: int = 0) -> list[list[str]]:
    """Синтетический корпус из 3 тем: животные / еда / транспорт.
    Предложение = 4–7 слов: слова одной темы + служебные слова («и», «в», «на», «очень», «где»)
    с большой частотой. Внутри темы слова часто стоят рядом, слова разных тем — никогда.
    В каждой теме 9 слов, и одно слово в каждой теме редкое (12 вхождений).
    Возвращает список предложений, каждое — список токенов."""
    rng = random.Random(seed)
    topic_names = list(TOPICS)
    sentences, sentence_topic = [], []
    for _ in range(n_sentences):
        topic = rng.choice(topic_names)
        common = [w for w in TOPICS[topic] if w != RARE[topic]]
        length = rng.randint(4, 7)
        n_topic = rng.randint(2, length - 1)  # минимум одно служебное слово
        words = rng.sample(common, n_topic) + rng.choices(FUNCTION_WORDS, k=length - n_topic)
        rng.shuffle(words)
        sentences.append(words)
        sentence_topic.append(topic)

    # редкое слово: в 12 предложениях своей темы заменяет одно тематическое слово
    for topic in topic_names:
        idx = [i for i, t in enumerate(sentence_topic) if t == topic]
        for i in rng.sample(idx, min(12, len(idx))):
            positions = [j for j, w in enumerate(sentences[i]) if w in TOPICS[topic]]
            sentences[i][rng.choice(positions)] = RARE[topic]
    return sentences


# ---------- пишу я ----------
def build_vocab(
    sentences: list[list[str]], min_count: int = 1
) -> tuple[dict[str, int], list[str], Int[Tensor, "V"]]:
    """Словарь по корпусу. Слова с частотой < min_count выкидываются.
    Возвращает:
        word2id: слово → id (0..V-1)
        id2word: список длины V, id2word[i] — слово с id i
        counts:  LongTensor (V,), counts[i] — сколько раз слово i встретилось в корпусе
    """
    ###
    uniq_word = []
    tf = {}

    for doc in sentences:
        for word in doc:
            if word in tf:
                tf[word] += 1
            else:
                tf[word] = 1

    for word, count in tf.items():
        if count < min_count:
            pass
        else:
            uniq_word.append(word)
    ###
    word2id = {}
    id2word = []

    for i, word in enumerate(uniq_word):
        word2id[word] = i
        id2word.append(word)
    ###
    counts = []

    for word in id2word:
        counts.append(tf[word])
    
    counts = torch.tensor(counts, dtype=torch.long)

    return word2id, id2word, counts


def skipgram_pairs(ids: list[int], window: int) -> list[tuple[int, int]]:
    """Обучающие пары Skip-Gram для ОДНОГО предложения (уже переведённого в id).
    Для каждой позиции t: пары (ids[t], ids[t+j]) для всех 1 ≤ |j| ≤ window, не выходя за края.
    Порядок: по t слева направо, внутри t — по j от -window до +window.
    Возвращает список пар (center_id, context_id).
    """
    pairs = []
    n = len(ids)

    for t in range(0, n):
        for j in range(-window, window+1):
            if j == 0:
                continue
            s = t + j
            if s < 0 or s >= n:
                continue
            pairs.append((ids[t], ids[s]))

    return pairs



def negative_distribution(counts: Int[Tensor, "V"], power: float = 0.75) -> Float[Tensor, "V"]:
    """Распределение, из которого тянем негативы: P(w) ∝ counts[w] ** power.
    Возвращает FloatTensor (V,), сумма = 1.
    """
    raise NotImplementedError


# ---------- готово: разобранный пример на незнакомое API (torch.multinomial) ----------
def sample_negatives(
    dist: Float[Tensor, "V"], batch: int, k: int, generator: torch.Generator | None = None
) -> Int[Tensor, "B K"]:
    """K случайных id на каждую из `batch` пар, по распределению dist (с возвращением)."""
    # torch.multinomial(input, num_samples, replacement, generator):
    #   input       — неотрицательные веса (V,); нормировать не обязательно, берутся пропорционально;
    #   num_samples — сколько индексов вытянуть;
    #   replacement — True: тянем с возвращением, один индекс может выпасть много раз
    #                 (без него нельзя вытянуть больше V индексов, а нам нужно B*K);
    #   generator   — свой ГСЧ: при одном и том же seed — те же индексы.
    # Результат — LongTensor (num_samples,) с индексами 0..V-1.
    flat = torch.multinomial(dist, batch * k, replacement=True, generator=generator)  # (B*K,): все негативы одним вызовом
    return flat.view(batch, k)  # (B*K,) -> (B, K): строка b — K негативов для пары b


class SGNS(nn.Module):
    """Skip-Gram with Negative Sampling.
    Два набора векторов: v (слово как центральное) и u (слово как контекстное).
    """

    def __init__(self, vocab_size: int, dim: int) -> None:
        """self.center:  nn.Embedding(V, d) — векторы v_w; инициализация U(-0.5/d, 0.5/d)
        self.context: nn.Embedding(V, d) — векторы u_w; инициализация нулями
        (так инициализирует оригинальный word2vec)."""
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        center: Int[Tensor, "B"],
        pos: Int[Tensor, "B"],
        neg: Int[Tensor, "B K"],
    ) -> Float[Tensor, ""]:
        """Средний по batch loss negative sampling:
            J = -log σ(u_pos · v_c) - Σ_{i=1..K} log σ(-u_neg_i · v_c)
        Возвращает скаляр (тензор формы ()).
        """
        raise NotImplementedError


def train(
    model: SGNS,
    pairs: Int[Tensor, "P 2"],
    dist: Float[Tensor, "V"],
    epochs: int,
    batch_size: int,
    k: int,
    lr: float,
    seed: int = 0,
) -> list[float]:
    """Обучение: каждую эпоху перемешать пары, идти батчами по batch_size,
    на каждый батч — свежие негативы (sample_negatives), шаг оптимизатора (Adam, lr).
    Возвращает список среднего loss за каждую эпоху (длина = epochs).
    """
    raise NotImplementedError


def most_similar(
    model: SGNS, word: str, word2id: dict[str, int], id2word: list[str], k: int = 5
) -> list[tuple[str, float]]:
    """k ближайших слов по КОСИНУСУ между центральными векторами (model.center).
    Само слово в ответ не входит. Сортировка по убыванию сходства.
    Возвращает [(слово, косинус), ...] длины k.
    """
    raise NotImplementedError
