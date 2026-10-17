# 02 — threading

`threading_tasks.py`: IO-задачи в пуле потоков (`fetch_all`), слияние результатов потоков без общего состояния (`count_words_parallel`) и потокобезопасный счётчик (`SafeCounter`). Сеть не нужна — «скачивание» имитируется `time.sleep`.
Тесты: `pytest -q` из этой папки.
`experiments.ipynb`: время последовательно vs потоки vs пул, и гонка `UnsafeCounter` vs `SafeCounter` вживую. Прогноз пишется до запуска, вывод — после.
