"""Блок 11 — multiprocessing: план чанков, подсчёт в процессах, сбор результатов."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# 1. План чанков
# ---------------------------------------------------------------------------
def make_chunks(items: list[T], n_chunks: int) -> list[list[T]]:
    """Делит items на n_chunks подряд идущих кусков почти равного размера.

    items:    N элементов
    n_chunks: K — на сколько кусков делить (K >= 1, иначе ValueError)
    return:   список кусков, где
              - склеить все куски по порядку == items (порядок и элементы сохранены)
              - размеры кусков отличаются максимум на 1, большие — первыми
                (N=10, K=3 → размеры [4, 3, 3])
              - пустых кусков нет: если N < K, кусков N (N=2, K=5 → [[a], [b]])
              - N = 0 → []
    """
    if n_chunks <= 0:
        raise ValueError("Количество чанков должно быть положительным")
    if len(items) == 0:
        return []
    if len(items) < n_chunks:
        return [[item] for item in items]

    min_group = len(items) // n_chunks
    residual = len(items) % n_chunks

    return [items[i*min_group + min(i, residual) : (i+1)*min_group + min(i+1, residual)] for i in range(n_chunks)]

    

# ---------------------------------------------------------------------------
# 2. Работа одного процесса
# ---------------------------------------------------------------------------
def count_words_chunk(texts: list[str]) -> Counter[str]:
    """Частоты слов по ПАЧКЕ текстов (выполняется внутри одного процесса).

    texts:  пачка из M текстов; слова текста — text.lower().split()
    return: Counter {слово: сколько раз встретилось во всей пачке}
    """
    result = Counter()

    for t in texts:
        t = t.lower().split()
        result.update(t)

    return result


# ---------------------------------------------------------------------------
# 3. Подсчёт в процессах + сбор результатов
# ---------------------------------------------------------------------------
def count_words_processes(texts: list[str], n_workers: int = 4) -> Counter[str]:
    """Частоты слов по ВСЕМ текстам, посчитанные в пуле процессов.

    План: разбить texts на n_workers пачек (make_chunks) → каждая пачка
    считается в своём процессе (count_words_chunk) → главный процесс
    сливает результаты в один Counter.

    texts:     N текстов
    n_workers: сколько процессов в пуле
    return:    Counter — то же, что последовательный count_words_chunk(texts)
    """
    chunks = make_chunks(texts, n_workers)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        result = Counter()

        words = pool.map(count_words_chunk, chunks)
        
        for process in words:
            result.update(process)

    return result


# --- для эксперимента 3 ---
COUNTER = 0


def bump(_: int) -> int:
    global COUNTER
    COUNTER += 1
    return COUNTER
