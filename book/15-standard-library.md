# Глава 15. Стандартная библиотека

Все модули стандартной библиотеки **вшиты в бинарник** (`stdlib/*.qlsp` генерируются в заголовки через `xxd -i` и компилируются в бинарник на этапе сборки — `cmake/stdlib_embed.cmake`). **11 модулей** загружаются **при старте** REPL/раннера (итерация по `embedded_modules[]` в `src/main.cpp:26-38`), причём порядок важен: `core` грузится первым (через `Interpreter::mark_module_loaded(m->name)`). `(import M)` — идиома для явности (повторный импорт — дешёвый no-op). С v2.3.0 поддержан частичный импорт: `(import-from (ml knn-predict nb-fit))`.

| Модуль | Файл | Размер | Содержание |
|---|---|---|---|
| `core` | `stdlib/core.qlsp` | 94 строки | утилиты списков, `let*`, тестовый фреймворк |
| `string` | `stdlib/string.qlsp` | (v2.3.0+) | строковые операции, char-коды, примитивы + композитные |
| `ml` | `stdlib/ml.qlsp` | 337 строк | классический ML: KNN, NB, Tree, RF, GB, KMeans, CV, GridSearch |
| `dl` | `stdlib/dl.qlsp` | 100+ строк (v2.3.0+) | слои-как-данные, `linear`, `net-forward`, `train-step`, `train-epochs` |
| `io` | `stdlib/io.qlsp` | 44 строки | файлы, CSV, `slurp`, `spit` |
| `regex` | `stdlib/regex.qlsp` | 17 строк | регулярки: `re-find`, `re-replace`, `re-split` |
| `audio` | `stdlib/audio.qlsp` | 101 строка | DSP: окна, синус, DFT, mel-шкала, dB |
| `datetime` | `stdlib/datetime.qlsp` | 40 строк | время: `now-ts`, `sleep`, `elapsed` |
| `pkg` | `stdlib/pkg.qlsp` | 77 строк | `pkg-install` через `http-get` |
| `errors` | `stdlib/errors.qlsp` | 64 строки | `make-error`, `throw`, `assert`, `try` |
| `visual` | `stdlib/visual.qlsp` | 33 строки | TensorBoard-подобный HTTP-дашборд |

> 🐍 «11 вшитых модулей в одном бинарнике» — это «один fat-пакет в одном бинарнике». Python-аналог: монолитный Docker-образ с предустановленными `numpy/torch/sklearn/tensorboard` — только без GIL и без 4 ГБ зависимостей.

## 15.1 core — всегда загружен

Утилиты списков (`length`, `reverse`, `member`, `nth`, `second`…), `let*`, тестовый фреймворк:

```lisp
(deftest "два-плюс-два"
  (assert-eq (+ 2 2) 4 "математика сломалась"))
(run-tests)
```

> 🐍 `pytest` в миниатюре: `deftest` ≈ `def test_*`, `run-tests` ≈ запуск pytest, `assert-eq` ≈ `assert a == ==`. Имена тестов склеиваются через `strcat` с префиксом `"test::"`.

Для тензоров есть `assert-zero` — проверка, что тензор почти нулевой (стандартная проверка в тестах градиентов):

```lisp
(defun assert-zero (tensor msg)
  (let ((s (tensor-scalar (TSUM (T* tensor tensor)))))
    (if (> (ABS s) 0.1)
      (progn (print "    FAIL:") (print msg) nil)
      t)))
```

Также определены 16 сокращений `car`/`cdr`: `caar`, `cadr`, `cdar`, `cddr`, `caaar`, `caadr`, `cadar`, `caddr`, `cdaar`, `cdadr`, `cddar`, `cdddr`, плюс `second`, `third`, `fourth`, `first`, `rest`.

## 15.2 string — строковые операции (v2.3.0+)

Новый модуль, зарегистрированный в `embedded_modules[]` сразу после `core`. Два слоя — примитивы и композитные функции.

### Примитивы (C++, регистрируются через `register_string_primitives`)

| Примитив | Что делает | Python-аналог |
|---|---|---|
| `STRING-LENGTH s` | длина строки | `len(s)` |
| `STRING-SPLIT s delim` | список подстрок | `s.split(delim)` |
| `STRING-JOIN list delim` | объединение | `delim.join(list)` |
| `STRING-UPCASE s` | uppercase | `s.upper()` |
| `STRING-DOWNCASE s` | lowercase | `s.lower()` |
| `STRING-TRIM s` | обрезка пробелов по краям | `s.strip()` |
| `STRING-REPLACE s old new` | замена всех вхождений | `s.replace(old, new)` |
| `STRING-INDEX s sub` | позиция подстроки или `nil` | `s.find(sub)` |
| `STRING-REVERSE s` | разворот | `s[::-1]` |
| `CHAR-CODE ch` | ASCII-код символа | `ord(ch)` |
| `CODE-CHAR n` | символ по ASCII-коду | `chr(n)` |
| `STRING-TO-NUMBER s` | парсинг числа | `float(s)` / `int(s)` |
| `STRING< a b` | лексикографическое `<` | `a < b` (для строк) |

### Композитные (на примитивах, в `stdlib/string.qlsp`)

```lisp
(import STRING)

(string-contains s sub)         ; STRING-INDEX, потом nil-check
(string-starts-with s prefix)
(string-ends-with s suffix)
(string-slice s start end)      ; подстрока
(string-lines s)                ; split по '\n'
(string-words s)                ; split по whitespace
(string-repeat s n)             ; n копий через STRING-JOIN
(to-string x)                   ; STRING(NUMBER-TO-STRING ...) для скаляров
(string s)                      ; константная строка-обёртка
```

> 🐍 Аналог модуля `str` Python (Python 3.6+). В Python многое встроено в сам язык (`len`, `+`, `[]`), здесь — отдельные примитивы из-за Lisp-стиля (один prim = одна функция).

### Замечание о `type-of`

С v2.3.0 `type-of` возвращает uppercase-имена типов:

| Значение | `type-of` | Python `type(x).__name__` |
|---|---|---|
| `42` | `INT` | `int` |
| `3.14` | `FLOAT` | `float` |
| `"hi"` | `STRING` | `str` |
| `'sym` | `SYMBOL` | (нет аналога) |
| `(1 2 3)` | `LIST` | `list` |
| `(tensor ...)` | `TENSOR` | `numpy.ndarray` / `torch.Tensor` |
| `(lambda ...)` / `(defun ...)` | `FUNCTION` | `function` |
| `#(...)` вектор | `VECTOR` | (нет аналога) |
| `nil` | `NIL` | `NoneType` |

> 🐍 В Python имена типов в lowercase, в QLISP — uppercase (после `intern`).

## 15.3 IO — файлы и строки

```lisp
(import IO)

(slurp "data.txt")                 ;; прочитать весь файл      open().read()
(spit "out.txt" "содержимое")      ;; записать                 open().write()
(file? "out.txt")                  ;; существует?              os.path.exists
(read-lines "data.txt")            ;; список строк             open().readlines()
(read-csv "table.csv")             ;; список списков полей     csv.reader (базовый)
```

Нижележащие примитивы (в `src/eval/interpreter.cpp`): `READ-FILE`, `WRITE-FILE`, `FILE-EXISTS?`.

## 15.4 REGEX — регулярные выражения

```lisp
(import REGEX)

(re-find "[0-9]+" "abc 123")        ;; первое совпадение        re.search(...).group()
(re-matches? "^[a-z]+$" "abc")      ;; bool                     bool(re.search(...))
(re-replace "[0-9]+" "N" "a1b22")   ;; замена                   re.sub
(re-split "," "a,b,c")              ;; разбиение                re.split (через \\x1F-маркер)
```

Нижележащие примитивы: `REGEX-MATCH`, `REGEX-REPLACE` (POSIX regex).

## 15.5 DATETIME — время

```lisp
(import DATETIME)

(now-ts)                 ;; unix-время (сек)          time.time()
(now-iso-str)            ;; ISO-строка                datetime.now().isoformat()
(sleep 2)                ;; пауза, сек                time.sleep
(wait-ms 500)            ;; пауза, мс                 time.sleep(0.5)
(timer-start)            ;; метка времени             t0 = time.time()
(timer-elapsed t0)       ;; сколько прошло            time.time() - t0
(seconds->hms 3665)      ;; → (1 1 5)

;; макрос замера времени
(elapsed (expensive-computation))    ;; напечатает elapsed: N s
```

> 🐍 `(elapsed expr)` ≈ contextmanager `timeit`/декоратор с замером времени. NB: `sleep` принимает секунды, `wait-ms` — миллисекунды.

## 15.6 AUDIO — DSP

```lisp
(import AUDIO)

(hann-window 1024)           ;; окно Ханна           scipy.signal.hann
(hamming-window 1024)        ;; окно Хэмминга
(sine-wave 440 2.0 44100)    ;; синус 440 Гц, 2 сек  np.sin(2πf t)
(fft-magnitudes signal)      ;; модули спектра       np.abs(np.fft.fft)
(stft signal 1024 256)       ;; оконное преобразование  librosa.stft
(hz-to-mel 440)              ;; мел-шкала            librosa.hz_to_mel
(mel-to-hz mel)
(lin-to-db amplitude)        ;; децибелы             20*log10
```

Примитив `FFT` (ядро) — быстрое преобразование Фурье Коули–Тьюки над тензором (см. `prim_fft` в `interpreter.cpp`).

`fft-magnitudes` возвращает список Python-стиля (не тензор) — для удобства работы с DS-пайплайнами.

## 15.7 ERRORS — обработка ошибок

```lisp
(import ERRORS)

(throw "что-то не так")            ;; бросить ошибку       raise
(assert condition "сообщение")     ;; упасть, если ложно   assert
(make-error 'type "детали")        ;; ошибочное значение-данные: (list 'error type msg)
(error-type e) (error-msg e)       ;; разбор
(type-check v 'float)              ;; проверка типа (int/float/string/list/tensor/nil)
(type-error? e) (value-error? e) (runtime-error? e)
(try (risky-op)
  (e (print (error-msg e))))       ;; try/catch
```

> 🐍 `try`-макрос ≈ `try/except e:`. Внутри `try` оборачивает body в `(lambda () body)` и вызывает через `catch-error`; ошибка связывается с символом из `catch-clause`.

## 15.8 VISUAL — дашборд обучения

```lisp
(import VISUAL)

(dashboard 8765)            ;; HTTP-сервер (TensorBoard-подобный) на порту 8765
(log-metric "loss" 0.25)    ;; скалярная метрика (авто-инкремент *step-counter*)
(log-tensor w "weights")    ;; heatmap тензора
(plot xs ys "loss-curve")   ;; линейный график
(scatter xs ys "points")    ;; scatter-плот
(imshow matrix "attention") ;; heatmap
(log-loss 0.1 5)            ;; shortcut: log-metric + явный step
```

Внутри — `start-dashboard`, `log-metric`, `log-tensor` (примитивы в `src/visual/dashboard.cpp`).

## 15.9 PKG — установка пакетов из GitHub

```lisp
(import PKG)
(pkg-install "user/repo")            ;; скачать pkg.qlsp из GitHub raw и установить
(pkg-install-branch "user/repo" "dev") ;; другая ветка
```

Внутри — `http-get` (примитив `prim_http_get`), `pkg-install-branch` идёт через `/tmp/qlisp-pkg-<name>/`, скачивает `pkg.qlsp` манифест и указанные в нём файлы через raw.githubusercontent.com. Подробнее — в главе 16.

## 15.10 IMPORT-FROM (v2.3.0+)

Частичный импорт: поднимает только перечисленные имена из модуля, не весь модуль.

```lisp
(import-from (ml knn-predict nb-fit))
;; после этого доступны только KNN-PREDICT и NB-FIT
;; остальное (KNN-FIT, TREE-FIT, RF-FIT, ...) остаётся недоступным
```

> 🐍 Аналог `from sklearn.neighbors import KNeighborsClassifier` в Python — но без скрытого `import sklearn` (модуль целиком не загружается как имя).

Реализация (`Interpreter::do_import_from`, `interpreter.cpp:1493`):

1. Загрузить модуль (вычислить его top-level формы).
2. Из всех `defvar`/`defun` модуля собрать список экспортов.
3. Из переданного списка имён проверить, что каждое определено в модуле (иначе — `runtime_error`).
4. Скопировать эти имена в текущий environment (через `set_global`).

Полезно в больших моделях — ускоряет загрузку и уменьшает namespace pollution.

## 15.11 BOUNDP (v2.3.0+)

```lisp
(boundp 'x)                ; T, если X связана в текущем/глобальном env
```

Полезно в макросах и meta-circular интерпретаторах: «определена ли переменная?» без чтения значения (т.е. без `eval`).

## 15.12 Сводка модулей

| Модуль | Назначение | Python-аналог |
|---|---|---|
| `core` | утилиты списков, `let*`, тесты | builtins + pytest |
| `string` | строки, char-коды | `str` (Python 3.6+) |
| `ml` | классический ML (337 строк на самом QLISP) | scikit-learn |
| `dl` | слои-как-данные, `linear`/`net-forward`/`train-step` | `torch.nn` (стиль Sequential) |
| `io` | файлы, CSV | `open`, `csv` |
| `regex` | регулярки | `re` |
| `audio` | DSP, окна, FFT, STFT, mel | `scipy.signal`, `librosa` |
| `datetime` | время | `time`, `datetime` |
| `pkg` | пакеты из GitHub | `pip` (мини) |
| `errors` | ошибки, try, assert | exceptions |
| `visual` | дашборд метрик | TensorBoard |