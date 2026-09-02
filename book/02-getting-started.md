# Глава 2. Начало работы

## 2.1 Установка

### Готовый бинарник (Linux)

```bash
curl -L https://github.com/FQingLars/QLISP/releases/latest/download/qlisp-linux-x86_64 -o qlisp
chmod +x qlisp && mv qlisp ~/.local/bin/
```

### Готовый бинарник (Windows)

Скачайте `qlisp-windows-x86_64.zip` со страницы релизов и распакуйте; `qlisp.exe`, `qlispc.exe`, `qlisp-lsp.exe`, `qvalent.exe` готовы к работе (MinGW-сборка, x86_64).

### Сборка из исходников

```bash
git clone https://github.com/FQingLars/QLISP.git && cd QLISP
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Требования: GCC 13+ / Clang 16+ (C++20), CMake 3.16+, **LLVM 22+** (для AOT-компиляции), OpenBLAS, OpenMP. GPU: CUDA Toolkit (опционально, код присутствует, но не тестирован).

Кросс-компиляция (в т.ч. Windows с Linux-машины через MinGW) описана в `README.md` исходного репозитория (`cmake/x86_64-w64-mingw32.cmake`).

## 2.2 Четыре инструмента

| Инструмент | Назначение | Python-аналог |
|---|---|---|
| `qlisp` | REPL + интерпретатор `.qlsp` | `python` |
| `qlispc` | Компилятор: `.qlsp` → LLVM IR → нативный бинарник | `cython`/`nuitka` |
| `qlisp-lsp` | Language Server (VS Code, neovim, Emacs) | pylance |
| `qvalent` | Пакетный менеджер (`qvalent.toml`) | `pip` |

Все четыре собираются из одних исходников (`CMakeLists.txt` → `BACKEND_SOURCES`); `qlisp-lsp` — отдельная цель только из `src/lsp/server.cpp` (без LLVM-зависимости).

## 2.3 REPL

```text
$ qlisp
QLISP
Enter expressions, press Ctrl+D to exit.

> (+ 1 2 3 4 5)
=> 15
> (defun square (x: Float) -> Float (* x x))
=> square
> (square 4.0)
=> 16.0
```

REPL при старте **автоматически загружает все 9 модулей стандартной библиотеки** (см. `embedded_modules[]` в `src/main.cpp`): `core`, `ml`, `io`, `regex`, `audio`, `datetime`, `pkg`, `errors`, `visual`. Они вшиты в бинарник через `cmake/stdlib_embed.cmake` (на этапе сборки `.qlsp` превращаются в `.h` через `xxd -i` и компилируются в бинарник).

> 🐍 **Python-аналогия.** Это как если бы `python` стартовал с предимпортированными `numpy`, `sklearn`, `re`, `datetime`, `time`, `urllib`, `tensorboard`.

Многострочный ввод работает по скобкам и кавычкам: незакрытый `(` или незакрытая `"` продолжают ввод.

После каждого выражения REPL вызывает `Interpreter::reset_scratch()` — disposable-область сбрасывается, буферы тензоров и cons-ячейки возвращаются в пулы (гл. 5).

## 2.4 Запуск файла

```lisp
;; hello.qlsp
(print "Hello, QLISP!")
(print (+ 40 2))
```

```text
$ qlisp hello.qlsp
Hello, QLISP!
42
:) | Successfully ran hello.qlsp
```

## 2.5 Импорт модулей

Модули стандартной библиотеки уже загружены в REPL, но в коде принято объявлять импорты явно (повторный импорт — дешёвый no-op; `Interpreter::mark_module_loaded()`):

```lisp
(import ML)        ;; классический ML: KNN, деревья, метрики, кросс-валидация
(import IO)        ;; файлы: slurp, spit, read-lines, read-csv
(import REGEX)     ;; регулярные выражения: re-find, re-replace, re-split
(import AUDIO)     ;; FFT, STFT, окна, mel-шкала, hz-to-mel
(import DATETIME)  ;; время: now-ts, now-iso-str, sleep, elapsed
(import PKG)       ;; пакетный менеджер: pkg-install из GitHub
(import ERRORS)    ;; ошибки: make-error, throw, assert, try
(import VISUAL)    ;; дашборд метрик (TensorBoard-подобный HTTP)
```

В исходниках есть также модуль `DL` (24 строки в `stdlib/dl.qlsp`: `linear`, `sequential`, `train-step`) — он вшит в бинарник наравне с остальными и доступен через `(import DL)`.

`(import M)` возвращает `"m-loaded"` и гарантирует, что модуль загружен однократно.

## 2.6 Компиляция

`qlispc` компилирует файл в LLVM IR, а затем (по умолчанию) в нативный бинарник:

```bash
# Нативный бинарник (LLVM IR → llc → .s → g++ -shared → dlopen)
qlispc hello.qlsp -o hello

# Только LLVM IR
qlispc compile --llvm hello.qlsp -o hello.ll
```

Полученный `hello` — самодостаточная программа, запускается без интерпретатора. Линковка идёт с `libqlisp_tensor_runtime.a` (тензорный рантайм, AVX2/FMA/OpenBLAS/OpenMP) и `libqlisp_runtime.a` (C-рантайм для AOT-линковки).

> 🐍 **Python-аналогия.** `qlispc file.qlsp -o prog` ≈ `nuitka --onefile file.py`. А HLO-компиляция тензорных функций (гл. 10) ≈ `torch.compile`, только путь — AOT через `llc` + `g++ -shared` + `dlopen`.

## 2.7 Редактор и LSP

`qlisp-lsp` — Language Server без внешних зависимостей (JSON-парсер встроен в `src/lsp/server.cpp`): автодополнения с hover-документацией, переход к определению для `defun`, `defvar`, `defmacro`, `defclass`, `defstruct`, `defhloop`, `defrule`.

- **VS Code**: любой LSP-клиент, команда запуска `qlisp-lsp`.
- **neovim**: `nvim-lspconfig` с custom config.
- **Emacs**: `lsp-mode` / `eglot`.

## 2.8 Первая ML-программа

```lisp
;; Первый градиентный спуск
(defvar x-data (tensor ((1.0) (2.0) (3.0) (4.0))))
(defvar y-data (tensor ((2.0) (4.0) (6.0) (8.0))))   ;; y = 2x

(defvar w (param (randn (list 1 1))))                ;; обучаемый вес
(defvar lr 0.01)

(defun train-step ()
  (let ((pred (matmul! x-data w)))     ;; forward
    (let ((loss (mse! pred y-data)))   ;; loss
      (grad! loss)                     ;; backward
      (sgd-step w lr)                  ;; обновить веса
      loss)))

(defvar epoch 0)
(while (< epoch 100)
  (let ((loss (train-step)))
    (if (equal (mod epoch 20) 0) (print loss)))
  (setq epoch (+ epoch 1)))
```

Не пугайтесь деталей — `param`, `matmul!`, `grad!` разобраны в главах 6–7. Полные рабочие примеры — в главе 17.

## 2.9 Куда смотреть дальше

- Синтаксис языка целиком — глава 3.
- Тензоры и автоград — главы 6–7.
- HLO-компиляция в нативный код — глава 10.
- Нейросимвольный слой — глава 14.
- Шпаргалка «QLISP ↔ Python» — глава 18.