# Session log — block11 / 02_threading

Старт: 2026-10-17 16:55

## Скелет
- Созданы `threading_tasks.py` (по handoff), `test_threading_tasks.py` (12 тестов), `experiments.ipynb` (10 ячеек, не запускался), `README.md`.
- `pytest -q` на скелете: 11 failed, 1 passed — проходит только `test_fetch_all_manual_order`, остальные падают на `NotImplementedError`.

## Попытки

### Попытка 1 (17:20) — задачи #1, #2
`pytest -q`: 6 failed, 6 passed (3 из failed — SafeCounter, ещё не начат).

- `fetch_all` — **все тесты прошли с первого раза** (порядок, пустой список, параллельность):
  ```python
  with ThreadPoolExecutor(max_workers=max_workers) as pool:
      result = list(pool.map(fetch, urls))
  return result
  ```
- `count_words` — `test_count_words_simple` упал: `AssertionError: assert Counter({' ':...': 1, 'r': 1}) == Counter({'cat...`
  ```python
  return Counter(text)
  ```
  Ошибка: считаются символы строки, а не слова; нет `.lower().split()`.
- `count_words_parallel` — `test_count_words_parallel_empty` и `..._matches_sequential` упали: `UnboundLocalError: cannot access local variable 'result'`
  ```python
  with ThreadPoolExecutor(max_workers=max_workers) as pool:
      for t in texts:
          result += count_words(t)
  return result
  ```
  Ошибки: (1) `result` не создан до `+=`; (2) пул создан, но ни одна задача в него не отправлена — `count_words` вызывается в главном потоке последовательно.

### Попытка 2 (17:41) — задача #2
`pytest -q`: 5 failed, 7 passed (3 из failed — SafeCounter).

- `count_words` — **исправлено, тесты зелёные**. Было → стало:
  ```python
  return Counter(text)
  # →
  text = text.lower().split()
  return Counter(text)
  ```
  Комментарий Вадима: «Думал, что это уже преобразованый текст без пробелов. в dock надо было так и писать прямо – не моя ошибка». (В докстринге было «Слова — text.lower().split().»)
- `count_words_parallel` — `..._matches_sequential`: `TypeError: 'Counter' object is not callable`; `..._empty`: `assert False  where False = isinstance([], Counter)`.
  ```python
  with ThreadPoolExecutor(max_workers=max_workers) as pool:
      result = list()
      for t in texts:
          result += pool.map(count_words(t), texts)
  return result
  ```
  Ошибки: (1) в `map` передан результат вызова `count_words(t)` (Counter), а не сама функция; (2) внешний цикл по `texts` + `map` по `texts` — каждый текст обрабатывался бы M раз; (3) `result` — `list`, а вернуть нужно `Counter`. Вадим: «вроде исправил, но не уверен».

### Попытка 3 (17:48) — задача #2
`pytest -q`: 4 failed, 8 passed (3 из failed — SafeCounter).

- `count_words_parallel` — `..._empty` теперь проходит (цикл не выполняется, возвращается пустой `Counter`); `..._matches_sequential`: `TypeError: 'ThreadPoolExecutor' object is not callable`.
  ```python
  result = list()
  for t in texts:
      result += pool.map(count_words(t), texts)
  # →
  result = Counter()
  for t in texts:
      result += pool(count_words, texts)
  ```
  Исправлено: тип `result` → `Counter`, в пул передаётся сама функция. Осталось: (1) пропал `.map` — вызывается сам объект пула; (2) внешний цикл по `texts` остался; (3) `map` возвращает итератор Counter'ов (по одному на текст), а не один Counter — `Counter += итератор` не сработает, их нужно сложить.

### Попытка 4 (17:51) — задача #2
`pytest -q`: 4 failed, 8 passed (3 из failed — SafeCounter).

- `count_words_parallel` — `..._matches_sequential`: `TypeError: unhashable type: 'Counter'`.
  ```python
  result = Counter()
  for t in texts:
      result += pool(count_words, texts)
  # →
  result = Counter(pool.map(count_words, texts))
  ```
  Исправлено: `.map` вернулся, внешний цикл убран. Осталось: `Counter(итерируемое)` считает **элементы** итерируемого — здесь элементы сами Counter'ы, их нельзя сделать ключами словаря. Нужно сложить M Counter'ов, а не посчитать их как элементы.

### Попытка 5 (17:59) — задача #2
`pytest -q`: 5 failed, 7 passed (3 из failed — SafeCounter).

- `count_words_parallel` — `..._matches_sequential`: `TypeError: unsupported operand type(s) for +=: 'NoneType' and 'Counter'`; `..._empty`: `assert False  where False = isinstance(None, Counter)`.
  ```python
  result = Counter(pool.map(count_words, texts))
  # →
  n = list(pool.map(count_words, texts))
  result = None
  for i in n:
      result += i
  ```
  Исправлено: Counter'ы теперь складываются в цикле. Осталось: начальное значение `result = None` — `None + Counter` не определено, а при пустом `texts` возвращается `None` вместо `Counter`.

### Попытка 6 (18:02) — задача #2
`pytest -q`: 3 failed, 9 passed (все 3 failed — SafeCounter, не начат).

- `count_words_parallel` — **все тесты зелёные**. Было → стало:
  ```python
  result = None
  # →
  result = Counter()
  ```
  Итоговый код:
  ```python
  with ThreadPoolExecutor(max_workers=max_workers) as pool:
      n = list(pool.map(count_words, texts))
      result = Counter()
      for i in n:
          result += i
  return result
  ```
  Задачи #1 и #2 закрыты (#2 — за 5 попыток после первой).

### Попытка 7 (18:15) — задача #3
`pytest -q`: 3 failed, 9 passed. Все 3 теста SafeCounter: `AttributeError: property 'value' of 'SafeCounter' object has no setter`.
```python
def __init__(self) -> None:
    self.value = 0

def increment(self, n: int = 1) -> None:
    lock = threading.Lock()
    with lock:
        self.value += n

@property
def value(self) -> int:
    return self.value
```
Ошибки: (1) хранилище названо так же, как property `value`: `self.value = 0` — попытка записать в property без setter, а `return self.value` внутри property вызывал бы сам себя бесконечно → нужен отдельный атрибут для числа; (2) `Lock` создаётся внутри `increment` — у каждого вызова свой новый замок, потоки друг друга не ждут, защиты нет → замок должен быть один на объект.

### Попытка 8 (18:23) — задача #3
`pytest -q`: 3 failed, 9 passed. Все 3 теста SafeCounter: `RecursionError: maximum recursion depth exceeded`.
```python
def __init__(self) -> None:
    self.value = 0
def increment(self, n: int = 1) -> None:
    lock = threading.Lock()
    with lock:
        self.value += n
# →
def __init__(self) -> None:
    self.num = self.value
    self.lock = threading.Lock()
def increment(self, n: int = 1) -> None:
    with self.lock:
        self.num += n
# property без изменений: return self.value
```
Исправлено: замок один на объект (создаётся в `__init__`), отдельный атрибут `num` для числа. Осталось: `self.num = self.value` — чтение property в `__init__`, а property возвращает `self.value` — вызывает сам себя → рекурсия. Направление связи перепутано: property должен читать `num`, а `num` — получить стартовое число.

### Попытка 9 (18:27) — задача #3
`pytest -q`: **12 passed** (прогнано 5 раз подряд — стабильно 12/12).
```python
self.num = self.value      →  self.num = 0
return self.value          →  return self.num
```
Итоговый код:
```python
def __init__(self) -> None:
    self.num = 0
    self.lock = threading.Lock()

def increment(self, n: int = 1) -> None:
    with self.lock:
        self.num += n

@property
def value(self) -> int:
    return self.num
```
Все три задачи закрыты, тесты зелёные. Дальше — ноутбук.

## Вопросы
Прямых вопросов не было. Реплики по ходу (дословно):
- к `count_words` (попытка 2): «Думал, что это уже преобразованый текст без пробелов. в dock надо было так и писать прямо – не моя ошибка»
- к `count_words_parallel` (попытка 2): «вроде исправил, но не уверен»
- перед экспериментами: «Выведи все эксперименты сюда. Чтобы я мог ответить голосом. Потом ты запишешь их. Самостоятельно.»

## Эксперименты
Вадим попросил вывести эксперименты в чат, ответил голосом; прогнозы/выводы вписывает в ноутбук и запускает его Claude.

### Эксперимент 1 — время (20 fetch по 0.3 с)
**Прогноз (дословно):**
> последовательно = 20 * 0,3 – просто последовательно добавляем ссылку, поток 1; fetch_all_manual = примерно 0,3, из-за того, что 20 потоков рабоют и каждый над своей ссылкой; fetch_all(max_workers=4) – не самый быстрый, потому что слишком мало потоков, обычно их десятки и подбирают эмпирически думаю по времени примерно будет 5 * 0.3; fetch_all(max_workers=20) – примерно будет 0,3 (тоже самое, что и fetch_all_manual – но если бы было больше ссылок, то чувствую, fetch_all_manual – проиграл)

**Результат (запуск 19:03):**
```
последовательно              6.09 c
fetch_all_manual             0.31 c
fetch_all(max_workers=4)     1.53 c
fetch_all(max_workers=20)    0.31 c
```

**Вывод (дословно):**
> ну в первом я был прав.

### Эксперимент 2 — гонка (8 потоков × 10 000, ожидается 80 000)
**Прогноз (дословно):**
> 1. Примерно на 20-25% меньше, из-за того, что 8 потоков параллельно будут пытаться менять одно и то же значений – в итоге, потеряют в определенный момент истиное значение (потоки вытесняют друг друга)
> 2. 80000
> 3. Одинаково: замок гарантирует тот факт, что поток будет внутри блока один + еще и там, и там одинаковое число потоков

**Результат (запуск 19:03):**
```
UnsafeCounter    10039 / 80000  потеряно:  69961  (0.21 c)
SafeCounter      80000 / 80000  потеряно:      0  (0.01 c)
```

**Вывод (дословно):**
> а во втором – по числу я был неправ, потому что интуиция подсказывает, что тут зависимость такая. У нас 80 тысяч, и 8 воркеров. Следовательно, все они пытаются работать параллельно, разделили между собой эту общую часть, и получилось ровно 10 тысяч. Ну, соответственно, из-за неправильной логики разделение это было неправильно. Именно поэтому это и получилось так. А со временем ошибся по интуиции, скорее всего, потому, что там внутри функции стоит еще sleep() Которая немного затормаживает работу.

## Итог
- `fetch_all` — с первого раза.
- `count_words` — со 2-й попытки: `Counter(text)` → `Counter(text.lower().split())` (считал символы, а не слова).
- `count_words_parallel` — с 6-й попытки. Ошибки по порядку: пул не использовался + `result` не создан → в `map` передан вызов `count_words(t)` вместо функции, лишний внешний цикл, `result = list()` → пропал `.map` (`pool(...)`) → `Counter(pool.map(...))` считал Counter'ы как элементы → `result = None` вместо `Counter()`.
- `SafeCounter` — с 3-й попытки. Ошибки: `Lock()` создавался внутри `increment` (у каждого вызова свой замок) + хранилище совпадало по имени с property `value` → `self.num = self.value` / `return self.value` (рекурсия) → `self.num = 0`, `return self.num`.
- Итог тестов: 12 passed (5 прогонов подряд).
- Эксперимент 1: все четыре прогноза совпали с фактом (6.09 / 0.31 / 1.53 / 0.31 c).
- Эксперимент 2: SafeCounter = 80 000 — совпало; UnsafeCounter = 10 039 при прогнозе «на 20–25% меньше» (факт ~87% потерь); время — прогноз «одинаково», факт 0.21 c vs 0.01 c.
- Время: 16:55 → 19:07, ≈ 2 ч 10 мин (включая паузы).
