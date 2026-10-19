"""Блок 11 — asyncio: конкурентные запросы, лимит одновременных запросов, отчёт по задачам."""
from __future__ import annotations

import asyncio
import time
from typing import TypedDict

# Счётчик «запросов в полёте» — по нему тесты проверяют, сколько запросов шло ОДНОВРЕМЕННО.
STATS: dict[str, int] = {"active": 0, "max_active": 0}


def reset_stats() -> None:
    STATS["active"] = 0
    STATS["max_active"] = 0


# ---------------------------------------------------------------------------
# 0. ГОТОВО: имитация запроса к API (сеть не нужна)
# ---------------------------------------------------------------------------
async def fake_fetch(item_id: int, delay: float) -> int:
    """Имитирует HTTP-запрос.

    item_id: id объекта (ids во всех задачах уникальны)
    delay:   сколько секунд «сервер» думает над ответом
    return:  item_id * 10
    raises:  ValueError, если item_id < 0 («сервер вернул ошибку»)
    """
    STATS["active"] += 1                                       # запрос ушёл — он «в полёте»
    STATS["max_active"] = max(STATS["max_active"], STATS["active"])
    try:
        await asyncio.sleep(delay)                             # ждём сеть: ЗДЕСЬ корутина отдаёт управление loop'у
        if item_id < 0:
            raise ValueError(f"bad id {item_id}")
        return item_id * 10
    finally:
        STATS["active"] -= 1                                   # выполняется всегда: успех, ошибка или отмена по таймауту


# ---------------------------------------------------------------------------
# 1. ГОТОВО (разобранный пример): последовательная версия — ловушка из претеста
# ---------------------------------------------------------------------------
async def fetch_one_by_one(ids: list[int], delays: list[float]) -> list[int]:
    """N запросов по очереди. Время ≈ sum(delays).

    ids:    N id
    delays: N задержек, delays[i] относится к ids[i]
    return: N значений в порядке ids
    """
    results: list[int] = []
    for item_id, delay in zip(ids, delays):
        value = await fake_fetch(item_id, delay)   # ждём ЭТОТ запрос до конца; корутина для следующего id ещё даже не создана,
        results.append(value)                      # поэтому loop'у не на что переключиться — он просто простаивает
    return results


# ---------------------------------------------------------------------------
# 2. ТВОЁ: все запросы конкурентно
# ---------------------------------------------------------------------------
async def fetch_all(ids: list[int], delays: list[float]) -> list[int]:
    """N запросов конкурентно. Время ≈ max(delays), а не sum(delays).

    ids:    N id
    delays: N задержек, delays[i] относится к ids[i]
    return: N значений в порядке ids (НЕ в порядке завершения)
    Если какой-то запрос упал — исключение пробрасывается наружу.
    """
    result = []
    for id, delay in zip(ids, delays):
        corrutine = fake_fetch(id, delay)
        result.append(corrutine)

    result = await asyncio.gather(*result)

    return result

    


# ---------------------------------------------------------------------------
# 3. ТВОЁ: конкурентно, но с лимитом одновременных запросов
# ---------------------------------------------------------------------------
async def fetch_all_limited(ids: list[int], delays: list[float], max_concurrent: int) -> list[int]:
    """Как fetch_all, но «в полёте» одновременно не больше max_concurrent запросов
    (зачем: у реальных API есть rate limit). Используй asyncio.Semaphore.

    ids, delays:    как в fetch_all
    max_concurrent: K >= 1, иначе ValueError
    return:         N значений в порядке ids
    Пример: 10 запросов по 0.1 с, K=2 → ≈ 0.5 с
    """
    if max_concurrent < 0.99:
        raise ValueError("Количество запросов должно быть больше 1")
    else:
        sem = asyncio.Semaphore(max_concurrent)

    async def semaph(sem, id, delay):
        async with sem:
            corrutine = await fake_fetch(id, delay)
            return corrutine

    tasks = []
    for id, delay in zip(ids, delays):
        task = semaph(sem, id, delay)
        tasks.append(task)

    final = await asyncio.gather(*tasks)

    return final


# ---------------------------------------------------------------------------
# 4. ТВОЁ: отчёт по задачам — ошибки и таймауты не ломают остальные запросы
# ---------------------------------------------------------------------------
class Report(TypedDict):
    ok: dict[int, int]        # id -> значение, для успешных запросов
    errors: dict[int, str]    # id -> имя класса исключения, например "ValueError"
    timeouts: list[int]       # id запросов, не уложившихся в timeout, в порядке ids
    elapsed: float            # секунды на весь вызов


async def fetch_report(ids: list[int], delays: list[float], timeout: float) -> Report:
    """N запросов конкурентно; каждый ограничен timeout секундами (asyncio.wait_for).
    Падение или таймаут одного запроса не должны отменять остальные.

    ids, delays: как в fetch_all
    timeout:     лимит на ОДИН запрос, сек
    return:      Report (см. класс выше)
    (Python >= 3.11: asyncio.TimeoutError — это встроенный TimeoutError)
    """
    t0 = time.perf_counter()                            # засечь время старта (как в тестах)

    # --- шаг 1: обёртка на ОДИН запрос (как semaph в #3) ---
    async def one(id, delay):
        try:
            value = await asyncio.wait_for(fake_fetch(id, delay), timeout)                  # fake_fetch(id, delay), обёрнутый в wait_for с timeout
            return ("ok", value)
        except TimeoutError:
            return ("timeout", None)
        except Exception as e:
            return ("error", type(e).__name__)              # имя класса исключения

    # --- шаг 2: N обёрток → один gather (как в #3) ---
    tasks = []
    for id, delay in zip(ids, delays):
        n = one(id, delay)
        tasks.append(n)

    outcomes = await asyncio.gather(*tasks)       # список из N пар ("ok"/"timeout"/"error", значение)   # в порядке ids — как всегда у gather

    
    # --- шаг 3: разложить исходы по отчёту ---
    report: Report = {"ok": {}, "errors": {}, "timeouts": [], "elapsed": 0.0}
    for id, (status, value) in zip(ids, outcomes):
        if status == "ok":
            report["ok"][id] = value                             # словарь: report["ok"][ключ] = значение
        elif status == "error":
            report["errors"][id] = value                            # report["errors"] — тоже словарь
        else:
            report["timeouts"].append(id)                            # report["timeouts"] — список, добавить id

    t1 = time.perf_counter()
    report["elapsed"] = t1-t0

    return report

    
