"""Блок 11 — threading: IO в пуле потоков, слияние результатов, безопасный счётчик."""
from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------------------
# 0. РАЗОБРАННЫЙ ПРИМЕР (готов, не трогать): то же самое «вручную» на Thread
# ---------------------------------------------------------------------------
def fetch_all_manual(urls: list[str], fetch: Callable[[str], str]) -> list[str]:
    """Запускает fetch(url) для каждого url в отдельном потоке.

    urls:  N адресов
    fetch: функция str -> str (IO: долго ждёт, почти не считает)
    return: список длины N, results[i] == fetch(urls[i]) — порядок как в urls
    """
    results: list[str | None] = [None] * len(urls)   # заранее N ячеек: каждый поток пишет в СВОЮ ячейку → гонки нет

    def worker(i: int, url: str) -> None:            # что выполнит один поток
        results[i] = fetch(url)                      # запись по своему индексу, общий только сам список

    threads = [
        threading.Thread(target=worker, args=(i, url))   # создать поток (ещё не запущен)
        for i, url in enumerate(urls)
    ]
    for t in threads:
        t.start()                                    # запустить все → ожидания fetch перекрываются
    for t in threads:
        t.join()                                     # дождаться КАЖДОГО, иначе results недозаполнен
    return results                                   # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 1. IO-задачи в пуле потоков
# ---------------------------------------------------------------------------
def fetch_all(urls: list[str], fetch: Callable[[str], str], max_workers: int = 8) -> list[str]:
    """То же, что fetch_all_manual, но через ThreadPoolExecutor.

    urls:        N адресов
    fetch:       функция str -> str
    max_workers: сколько потоков в пуле одновременно
    return:      список длины N, results[i] == fetch(urls[i]) — порядок как в urls,
                 даже если потоки закончили в другом порядке
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        result = list(pool.map(fetch, urls))

    return result



# ---------------------------------------------------------------------------
# 2. Слияние результатов потоков (без общего изменяемого состояния)
# ---------------------------------------------------------------------------
def count_words(text: str) -> Counter[str]:
    """Частоты слов ОДНОГО текста. Слова — text.lower().split().

    return: Counter {слово: сколько раз встретилось в text}
    """
    text = text.lower().split()
    return Counter(text)


def count_words_parallel(texts: list[str], max_workers: int = 8) -> Counter[str]:
    """Частоты слов по ВСЕМ текстам, тексты обрабатываются в пуле потоков.

    Правило: потоки НЕ пишут в общий словарь. Каждый поток возвращает свой результат,
    итог собирает главный поток.

    texts:  M текстов
    return: Counter — то же, что посчитать все тексты последовательно
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        n = list(pool.map(count_words, texts))
        result = Counter()

        for i in n:
            result += i

    return result



# ---------------------------------------------------------------------------
# 3. Потокобезопасный счётчик
# ---------------------------------------------------------------------------
class SafeCounter:
    """Счётчик, который можно увеличивать из многих потоков без потерь."""

    def __init__(self) -> None:
        self.num = 0
        self.lock = threading.Lock()

    def increment(self, n: int = 1) -> None:
        """Увеличить значение на n. Должно быть безопасно из любого числа потоков."""
        with self.lock:
            self.num += n

    @property
    def value(self) -> int:
        """Текущее значение (int)."""
        return self.num