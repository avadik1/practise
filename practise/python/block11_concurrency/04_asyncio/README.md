# 04 — asyncio

`async_tasks.py`: имитация запроса (`fake_fetch`), последовательная версия (`fetch_one_by_one`), конкурентная (`fetch_all`), с лимитом через `Semaphore` (`fetch_all_limited`) и отчёт с ошибками/таймаутами (`fetch_report`).
Тесты: `pytest -q` из этой папки (обычные sync-тесты, внутри `asyncio.run(...)`, без pytest-asyncio).
`experiments.ipynb`: `for … await` против конкурентного запуска, блокирующий вызов в event loop и `asyncio.to_thread`.
В Jupyter вместо `asyncio.run(...)` пишут `await ...` прямо в ячейке: event loop там уже запущен, и `asyncio.run` даёт `RuntimeError`.
