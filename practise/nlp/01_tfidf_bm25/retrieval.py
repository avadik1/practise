"""Поисковый baseline: TF-IDF и BM25 с нуля.

Обозначения размерностей:
    N — число документов в корпусе
    V — размер словаря (число уникальных токенов корпуса)
    n — число текстов, переданных в transform
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from jaxtyping import Float

TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")  # как в sklearn: слова от 2 символов


def tokenize(text: str) -> list[str]:
    """ГОТОВО. Нижний регистр + слова от 2 символов.
    'Код 0x80070057, я!' -> ['код', '0x80070057']
    """
    return TOKEN_RE.findall(text.lower())


def top_k(scores: Float[np.ndarray, "N"], k: int) -> list[tuple[int, float]]:
    """ГОТОВО. k лучших документов по убыванию скора: [(doc_id, score), ...]."""
    k = min(k, len(scores))
    idx = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in idx]


class TfidfRetriever:
    """TF-IDF как в sklearn.TfidfVectorizer() с параметрами по умолчанию.

    tf(t, d) = сколько раз токен t встретился в документе d (сырой счётчик)
    df(t)    = в скольких документах корпуса есть t
    idf(t)   = ln((1 + N) / (1 + df(t))) + 1
    вектор d = [tf(t, d) * idf(t) для каждого t словаря], затем L2-нормировка строки

    Отличие от «учебной» формулы log(N/df): +1 внутри — защита от деления на 0,
    +1 снаружи — слово, которое есть во всех документах, не обнуляется полностью.
    """

    def fit(self, docs: list[str]) -> "TfidfRetriever":
        """Учит словарь и idf по корпусу из N документов.

        Сохраняет:
            self.vocab_:      dict[str, int] — токен -> номер столбца;
                              номера присваиваются по алфавиту токенов (как в sklearn)
            self.idf_:        Float[np.ndarray, "V"]
            self.doc_matrix_: Float[np.ndarray, "N V"] — векторы корпуса для поиска
        Возвращает self.
        """
        ###1
        self.vocab_ = {}

        words = list()

        for w in docs:
            w = tokenize(w)
            words += w

        words = set(words)
        words = sorted(words)

        for i, word in enumerate(words):
            self.vocab_[word] = i
        ###2
        self.idf_ = np.zeros(len(words))

        df = {}
        df_docs_words = list()

        for doc in docs:
            doc = tokenize(doc)
            df_docs_words.append(doc)

        for doc in df_docs_words:
            doc = set(doc)
            for word in doc:
                if word in df:
                    df[word] += 1
                else:
                    df[word] = 1
        ###
        for word, i in self.vocab_.items():
            self.idf_[i] = np.log((1 + len(docs)) / (1 + df[word])) + 1
        ###3
        self.doc_matrix_ = self.transform(docs)

        return self
    
    def transform(self, texts: list[str]) -> Float[np.ndarray, "n V"]:
        """Каждый текст -> L2-нормированный TF-IDF вектор длины V.

        Токены вне словаря игнорируются.
        Текст без единого известного токена -> нулевая строка (без NaN).
        """
        ###
        tf_docs = list()

        for doc in texts:
            doc = tokenize(doc)
            tf_docs.append(doc)
        ###
        tf_df = np.zeros((len(texts), len(self.vocab_)))

        for i, doc in enumerate(tf_docs):
            for word, count in Counter(doc).items():
                if word in self.vocab_:
                    tf_df[i, self.vocab_[word]] = (count * self.idf_[self.vocab_[word]])

        return (tf_df / (np.linalg.norm(tf_df, axis=1, keepdims=True) + 1e-7))

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        """Top-k документов по косинусному сходству с запросом."""
        v = self.transform([query])
        cosins = self.doc_matrix_ @ v[-1]

        return top_k(cosins, k)


class BM25:
    """Okapi BM25.

    score(q, d) = сумма по УНИКАЛЬНЫМ токенам t запроса:
        idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))   # вариант Lucene, всегда > 0
    tf     = сколько раз t встретился в документе d
    |d|    = длина d в токенах, avgdl — средняя длина документа корпуса
    Токен запроса, которого нет в корпусе, даёт вклад 0.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def fit(self, docs: list[str]) -> "BM25":
        """Сохраняет:
            self.doc_tfs_: list[Counter] — длины N: счётчик токенов каждого документа
            self.doc_len_: Float[np.ndarray, "N"] — длины документов в токенах
            self.avgdl_:   float
            self.idf_:     dict[str, float] — idf для каждого токена корпуса
        Возвращает self.
        """
        raise NotImplementedError

    def score(self, query: str) -> Float[np.ndarray, "N"]:
        """BM25-скор запроса для каждого из N документов корпуса."""
        raise NotImplementedError

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        """Top-k документов по BM25-скору."""
        raise NotImplementedError
