import math

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from retrieval import BM25, TfidfRetriever, tokenize, top_k

DOCS = [
    "Как оформить возврат товара и вернуть деньги",
    "Доставка курьером по Москве занимает два дня",
    "Оплата картой или наличными при получении заказа",
    "Гарантия на технику один год, ремонт в сервисном центре",
    "Курьер привезёт заказ, оплата картой курьеру",
]


# ---------- tokenize / top_k ----------

def test_tokenize_docstring_example():
    """tokenize: нижний регистр, слова короче 2 символов выкидываются."""
    assert tokenize("Код 0x80070057, я!") == ["код", "0x80070057"]


def test_top_k_descending_and_capped():
    """top_k: по убыванию скора и не больше len(scores)."""
    scores = np.array([0.1, 0.9, 0.5])
    assert top_k(scores, 2) == [(1, 0.9), (2, 0.5)]
    assert len(top_k(scores, 10)) == 3


# ---------- TfidfRetriever: сверка со sklearn ----------

@pytest.fixture
def fitted():
    return TfidfRetriever().fit(DOCS), TfidfVectorizer().fit(DOCS)


def test_tfidf_vocab_matches_sklearn(fitted):
    """Словарь (токен -> столбец) совпадает со sklearn: те же токены, нумерация по алфавиту."""
    mine, ref = fitted
    assert mine.vocab_ == ref.vocabulary_


def test_tfidf_idf_matches_sklearn(fitted):
    """idf_ совпадает со sklearn (сглаженная формула ln((1+N)/(1+df)) + 1)."""
    mine, ref = fitted
    assert np.allclose(mine.idf_, ref.idf_)


def test_tfidf_shapes(fitted):
    """Формы: idf_ — (V,), doc_matrix_ — (N, V)."""
    mine, ref = fitted
    V = len(ref.vocabulary_)
    assert mine.idf_.shape == (V,)
    assert mine.doc_matrix_.shape == (len(DOCS), V)


def test_tfidf_transform_corpus_matches_sklearn(fitted):
    """transform(корпус) совпадает со sklearn: tf * idf + L2-нормировка строк."""
    mine, ref = fitted
    assert np.allclose(mine.transform(DOCS), ref.transform(DOCS).toarray())


def test_tfidf_transform_new_texts_matches_sklearn(fitted):
    """transform новых текстов: незнакомые слова игнорируются, результат как у sklearn."""
    mine, ref = fitted
    texts = ["вернуть деньги за наушники", "курьер курьер оплата картой срочно"]
    assert np.allclose(mine.transform(texts), ref.transform(texts).toarray())


def test_tfidf_unknown_only_text_is_zero_row(fitted):
    """Текст только из незнакомых слов -> нулевая строка, без NaN (нет деления на 0)."""
    mine, _ = fitted
    row = mine.transform(["абракадабра шмяк"])
    assert row.shape == (1, len(mine.vocab_))
    assert not np.isnan(row).any()
    assert np.all(row == 0)


def test_tfidf_search_exact_doc_first(fitted):
    """search: запрос = текст документа -> этот документ первый, длина = k, скоры по убыванию."""
    mine, _ = fitted
    res = mine.search(DOCS[2], k=3)
    assert len(res) == 3
    assert res[0][0] == 2
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)


# ---------- BM25 ----------

def bm25_term(tf, dl, avgdl, N, df, k1=1.5, b=0.75):
    """Эталон: явная формула вклада одного термина."""
    idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))


def test_bm25_single_term_manual():
    """Скор одного термина на корпусе из 3 документов совпадает с ручным расчётом."""
    docs = ["кот кот пёс", "пёс мышь", "кот рыба рыба рыба"]
    # длины 3, 2, 4 -> avgdl = 3; 'кот' есть в 2 документах из 3
    s = BM25().fit(docs).score("кот")
    expected = [bm25_term(2, 3, 3.0, 3, 2), 0.0, bm25_term(1, 4, 3.0, 3, 2)]
    assert np.allclose(s, expected)


def test_bm25_score_shape():
    """score возвращает массив формы (N,)."""
    s = BM25().fit(DOCS).score("заказ")
    assert s.shape == (len(DOCS),)


def test_bm25_idf_positive_for_everywhere_word():
    """idf слова, которое есть во всех документах, > 0 (Lucene-вариант не уходит в минус)."""
    m = BM25().fit(["кот спит", "кот ест", "кот играет"])
    assert m.idf_["кот"] > 0


def test_bm25_tf_saturation():
    """Насыщение: tf = 1, 2, 3 при одинаковой длине -> скор растёт, приросты убывают."""
    docs = [
        "ключ аа бб вв",
        "ключ ключ аа бб",
        "ключ ключ ключ аа",
        "гг дд ее жж",
    ]
    s = BM25().fit(docs).score("ключ")
    assert s[0] < s[1] < s[2]
    assert (s[1] - s[0]) > (s[2] - s[1])


def test_bm25_k1_zero_ignores_tf():
    """k1 = 0 -> tf не важен: документы с tf = 1 и tf = 5 одинаковой длины равны;
    документ без термина -> 0, а не NaN (0 / 0)."""
    docs = [
        "ключ аа бб вв гг",
        "ключ ключ ключ ключ ключ",
        "дд ее жж зз ии",
    ]
    s = BM25(k1=0).fit(docs).score("ключ")
    assert s[0] > 0
    assert np.isclose(s[0], s[1])
    assert s[2] == 0


def test_bm25_b_controls_length_normalization():
    """b = 0 -> длина не важна (одинаковый tf = одинаковый скор); b = 0.75 -> короткий выше."""
    docs = [
        "ключ аа",
        "ключ бб вв гг дд ее жж зз",
        "ии кк",
    ]
    s0 = BM25(b=0).fit(docs).score("ключ")
    assert np.isclose(s0[0], s0[1])
    s75 = BM25(b=0.75).fit(docs).score("ключ")
    assert s75[0] > s75[1]


def test_bm25_unknown_query_zeros():
    """Запрос только из незнакомых слов -> все скоры 0."""
    s = BM25().fit(DOCS).score("абракадабра шмяк")
    assert np.all(s == 0)


def test_bm25_duplicate_query_tokens():
    """Запрос «слово слово» = запрос «слово»: считаются уникальные токены запроса."""
    m = BM25().fit(DOCS)
    assert np.allclose(m.score("заказ заказ"), m.score("заказ"))
