# Глава 15. Стандартная библиотека

Все модули стандартной библиотеки **вшиты в бинарник** (`stdlib/*.qlsp` компилируются в бинарник на этапе сборки). Ядро (`core`) загружается всегда, остальные — по `(import ...)`.

## 15.1 core — всегда загружен

Утилиты списков (`length`, `reverse`, `member`, `nth`, `second`…), `let*`, тестовый фреймворк:

```lisp
(deftest "два-плюс-два"
  (assert-eq (+ 2 2) 4 "математика сломалась"))
(run-tests)
```

> 🐍 `pytest` в миниатюре: `deftest` ≈ `def test_*`, `run-tests` ≈ запуск pytest, `assert-eq` ≈ `assert a == b`.

Для тензоров есть `assert-zero (gradient)` — проверка, что тензор почти нулевой (стандартная проверка в тестах градиентов).

## 15.2 IO — файлы и строки

```lisp
(import IO)

(slurp "data.txt")                 ;; прочитать весь файл      open().read()
(spit "out.txt" "содержимое")      ;; записать                 open().write()
(file? "out.txt")                  ;; существует?              os.path.exists
(read-lines "data.txt")            ;; список строк             open().readlines()
(read-csv "table.csv")             ;; список списков полей     csv.reader
```

Нижележащие примитивы: `READ-FILE`, `WRITE-FILE`, `FILE-EXISTS?`.

## 15.3 REGEX — регулярные выражения

```lisp
(import REGEX)

(re-find "[0-9]+" "abc 123")        ;; первое совпадение        re.search(...).group()
(re-matches? "^[a-z]+$" "abc")      ;; bool                     bool(re.search(...))
(re-replace "[0-9]+" "N" "a1b22")   ;; замена                   re.sub
(re-split "," "a,b,c")              ;; разбиение                re.split
```

## 15.4 DATETIME — время

```lisp
(import DATETIME)

(now-ts)                 ;; unix-время (сек)          time.time()
(now-iso)                ;; ISO-строка                datetime.now().isoformat()
(sleep 2)                ;; пауза, сек                time.sleep
(sleep-ms 500)           ;; пауза, мс
(timer-start)            ;; метка времени             t0 = time.time()
(timer-elapsed t0)       ;; сколько прошло            time.time() - t0
(seconds->hms 3665)      ;; → (1 1 5)

;; макрос замера времени
(elapsed (expensive-computation))    ;; напечатает elapsed: N s
```

> 🐍 `(elapsed expr)` ≈ contextmanager `timeit`/декоратор с замером времени.

## 15.5 AUDIO — DSP

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

Примитив `FFT` (ядро) — быстрое преобразование Фурье Коули–Тьюки над тензором.

## 15.6 ERRORS — обработка ошибок

```lisp
(import ERRORS)

(throw "что-то не так")            ;; бросить ошибку       raise
(assert condition "сообщение")     ;; упасть, если ложно   assert
(make-error 'type "детали")        ;; ошибочное значение-данные
(error-type e) (error-msg e)       ;; разбор
(type-check v 'float)              ;; проверка типа

(try (risky-op)
  (e (print (error-msg e))))       ;; try/catch
```

> 🐍 `try`-макрос ≈ `try/except e:`.

## 15.7 VISUAL — дашборд обучения

```lisp
(import VISUAL)

(dashboard 8765)            ;; HTTP-сервер (TensorBoard-подобный)
(log-metric "loss" 0.25)    ;; скалярная метрика (авто-шаг)
(log-tensor w "weights")    ;; heatmap тензора
(plot xs ys "loss-curve")   ;; линейный график
(scatter xs ys "points")    ;; scatter-плот
(imshow matrix "attention") ;; heatmap
(log-loss 0.1 5)            ;; shortcut: log-metric + step
```

## 15.8 PKG — установка пакетов из GitHub

```lisp
(import PKG)
(pkg-install "user/repo")       ;; скачать pkg.qlsp из GitHub raw и установить
```

Подробнее — в главе 16.

## 15.9 Сводка модулей

| Модуль | Назначение | Python-аналог |
|---|---|---|
| `core` | утилиты списков, `let*`, тесты | builtins + pytest |
| `ml` | классический ML | scikit-learn |
| `dl` | слои и модели | torch.nn |
| `io` | файлы, CSV | open, csv |
| `regex` | регулярки | re |
| `audio` | DSP, окна, FFT, STFT, mel | scipy.signal, librosa |
| `datetime` | время | time, datetime |
| `pkg` | пакеты из GitHub | pip (мини) |
| `errors` | ошибки, try, assert | exceptions |
| `visual` | дашборд метрик | TensorBoard |
