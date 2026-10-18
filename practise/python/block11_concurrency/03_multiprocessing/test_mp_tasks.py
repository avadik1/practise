import random
from collections import Counter

import pytest

from mp_tasks import count_words_chunk, count_words_processes, make_chunks


def make_texts(n: int = 500, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(200)] + ["Cat", "CAT", "dog", "Dog"]
    return [" ".join(rng.choices(vocab, k=rng.randint(0, 30))) for _ in range(n)]


# --- 1. make_chunks ----------------------------------------------------------
def test_make_chunks_10_3():
    items = list(range(10))
    chunks = make_chunks(items, 3)
    assert [len(c) for c in chunks] == [4, 3, 3]
    assert [x for c in chunks for x in c] == items


def test_make_chunks_even():
    items = list(range(9))
    assert make_chunks(items, 3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_make_chunks_fewer_items_than_chunks():
    assert make_chunks(["a", "b"], 5) == [["a"], ["b"]]


def test_make_chunks_empty():
    assert make_chunks([], 3) == []


def test_make_chunks_single_chunk():
    items = list("hello")
    assert make_chunks(items, 1) == [items]


@pytest.mark.parametrize("k", [0, -1])
def test_make_chunks_bad_k(k):
    with pytest.raises(ValueError):
        make_chunks([1, 2, 3], k)


@pytest.mark.parametrize("n", [0, 1, 2, 5, 7, 10, 16, 33, 100])
@pytest.mark.parametrize("k", [1, 2, 3, 4, 7, 8, 50])
def test_make_chunks_properties(n, k):
    items = list(range(n))
    chunks = make_chunks(items, k)
    assert [x for c in chunks for x in c] == items
    assert len(chunks) == min(n, k)
    assert all(len(c) > 0 for c in chunks)
    if chunks:
        sizes = [len(c) for c in chunks]
        assert max(sizes) - min(sizes) <= 1
        assert sizes == sorted(sizes, reverse=True)


# --- 2. count_words_chunk ----------------------------------------------------
def test_count_words_chunk_simple():
    texts = ["The cat sat", "the CAT ran", "a dog"]
    assert count_words_chunk(texts) == Counter(
        {"the": 2, "cat": 2, "sat": 1, "ran": 1, "a": 1, "dog": 1}
    )


def test_count_words_chunk_empty():
    assert count_words_chunk([]) == Counter()


# --- 3. count_words_processes ------------------------------------------------
def test_count_words_processes_matches_sequential():
    texts = make_texts()
    assert count_words_processes(texts, n_workers=4) == count_words_chunk(texts)


def test_count_words_processes_workers_1_vs_8():
    texts = make_texts(seed=1)
    assert count_words_processes(texts, n_workers=1) == count_words_processes(
        texts, n_workers=8
    )


def test_count_words_processes_empty():
    assert count_words_processes([], n_workers=4) == Counter()
