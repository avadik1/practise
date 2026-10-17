import random
import threading
import time
from collections import Counter

from threading_tasks import (
    SafeCounter,
    count_words,
    count_words_parallel,
    fetch_all,
    fetch_all_manual,
)


def slow_upper(url: str) -> str:
    time.sleep(random.uniform(0.05, 0.2))
    return url.upper()


def make_urls(n: int = 20) -> list[str]:
    return [f"http://site{i}.com/page" for i in range(n)]


# --- 0. разобранный пример ---------------------------------------------------
def test_fetch_all_manual_order():
    urls = make_urls()
    assert fetch_all_manual(urls, slow_upper) == [u.upper() for u in urls]


# --- 1. fetch_all ------------------------------------------------------------
def test_fetch_all_order():
    urls = make_urls()
    assert fetch_all(urls, slow_upper, max_workers=8) == [u.upper() for u in urls]


def test_fetch_all_empty():
    assert fetch_all([], slow_upper) == []


def test_fetch_all_is_parallel():
    def fetch(url: str) -> str:
        time.sleep(0.2)
        return url

    urls = make_urls(16)
    start = time.perf_counter()
    result = fetch_all(urls, fetch, max_workers=16)
    elapsed = time.perf_counter() - start
    assert result == urls
    assert elapsed < 1.0, f"{elapsed:.2f} c — похоже, вызовы идут последовательно (было бы 3.2 c)"


# --- 2. count_words ----------------------------------------------------------
def test_count_words_simple():
    assert count_words("Cat dog cat\nDOG cat bird") == Counter({"cat": 3, "dog": 2, "bird": 1})


def test_count_words_returns_counter():
    assert isinstance(count_words("a b a"), Counter)


def test_count_words_empty():
    assert count_words("") == Counter()


def test_count_words_parallel_matches_sequential():
    rng = random.Random(0)
    vocab = ["alpha", "Beta", "GAMMA", "delta", "eps", "Zeta", "eta", "theta"]
    texts = [" ".join(rng.choices(vocab, k=rng.randint(0, 50))) for _ in range(200)]
    expected = sum((count_words(t) for t in texts), Counter())
    assert count_words_parallel(texts, max_workers=8) == expected


def test_count_words_parallel_empty():
    result = count_words_parallel([])
    assert isinstance(result, Counter)
    assert result == Counter()


# --- 3. SafeCounter ----------------------------------------------------------
def test_safe_counter_starts_at_zero():
    assert SafeCounter().value == 0


def test_safe_counter_increment_n():
    c = SafeCounter()
    c.increment(5)
    assert c.value == 5
    c.increment()
    assert c.value == 6


def test_safe_counter_many_threads():
    c = SafeCounter()

    def work() -> None:
        for _ in range(50_000):
            c.increment()

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.value == 400_000
