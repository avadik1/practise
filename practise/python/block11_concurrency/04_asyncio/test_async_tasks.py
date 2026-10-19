import asyncio
import time

import pytest

from async_tasks import STATS, fetch_all, fetch_all_limited, fetch_one_by_one, fetch_report, reset_stats


@pytest.fixture(autouse=True)
def _reset():
    reset_stats()


def timed(coro):
    t0 = time.perf_counter()
    result = asyncio.run(coro)
    return result, time.perf_counter() - t0


# --- 1. fetch_one_by_one (готовый пример) -------------------------------------
def test_one_by_one():
    result, elapsed = timed(fetch_one_by_one([1, 2, 3], [0.05] * 3))
    assert result == [10, 20, 30]
    assert elapsed >= 0.15


# --- 2. fetch_all --------------------------------------------------------------
def test_fetch_all_order():
    result, _ = timed(fetch_all([1, 2, 3], [0.3, 0.2, 0.1]))
    assert result == [10, 20, 30]


def test_fetch_all_speed():
    _, elapsed = timed(fetch_all(list(range(20)), [0.2] * 20))
    assert elapsed < 0.5


def test_fetch_all_concurrency():
    timed(fetch_all(list(range(20)), [0.2] * 20))
    assert STATS["max_active"] == 20


def test_fetch_all_empty():
    result, _ = timed(fetch_all([], []))
    assert result == []


def test_fetch_all_raises():
    with pytest.raises(ValueError):
        asyncio.run(fetch_all([1, -1], [0.05, 0.05]))


# --- 3. fetch_all_limited ------------------------------------------------------
def test_limited_order():
    result, _ = timed(fetch_all_limited([1, 2, 3], [0.3, 0.2, 0.1], 2))
    assert result == [10, 20, 30]


def test_limited_limit():
    _, elapsed = timed(fetch_all_limited(list(range(10)), [0.1] * 10, 2))
    assert 0.45 <= elapsed < 0.8
    assert STATS["max_active"] == 2


def test_limited_k_ge_n():
    timed(fetch_all_limited([1, 2, 3], [0.05] * 3, 10))
    assert STATS["max_active"] == 3


def test_limited_k_zero():
    with pytest.raises(ValueError):
        asyncio.run(fetch_all_limited([1, 2], [0.05, 0.05], 0))


# --- 4. fetch_report -----------------------------------------------------------
MIX_IDS = [1, -2, 3, 4]
MIX_DELAYS = [0.1, 0.1, 1.0, 0.1]


def test_report_mix():
    report = asyncio.run(fetch_report(MIX_IDS, MIX_DELAYS, 0.3))
    assert report["ok"] == {1: 10, 4: 40}
    assert report["errors"] == {-2: "ValueError"}
    assert report["timeouts"] == [3]
    assert report["elapsed"] < 0.6


def test_report_all_ok():
    report = asyncio.run(fetch_report([1, 2], [0.05, 0.05], 1))
    assert report["ok"] == {1: 10, 2: 20}
    assert report["errors"] == {}
    assert report["timeouts"] == []


def test_report_cleanup():
    asyncio.run(fetch_report(MIX_IDS, MIX_DELAYS, 0.3))
    assert STATS["active"] == 0
