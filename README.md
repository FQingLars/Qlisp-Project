# 📖 QLISP Book

**Ориентировочная книга по языку программирования QLISP** — компилируемому Лиспу для машинного обучения и AI.

QLISP — это диалект Лиспа, который компилируется в нативный машинный код через LLVM, со встроенной тензорной системой (F16-поддержка с v2.2.0), автоградиентом, HLO-компиляцией с AOT-инференсом и нейросимвольным движком (rule engine, унификация, `NS-IF`/`NS-GRAD!` с v2.2.0+) — всё в стандартной библиотеке, без единого `pip install`.

Книга написана для **инженеров с Python-бэкграундом**: каждая глава содержит аналогии с Python, NumPy, PyTorch и scikit-learn.

## Оглавление

| Глава | Тема | Python-аналог |
|-------|------|---------------|
| [01. Введение](book/01-introduction.md) | Зачем нужен QLISP, философия, пайплайн | «Почему не Python?» |
| [02. Начало работы](book/02-getting-started.md) | Установка, REPL, первая программа, компиляция | `python`, `pip` |
| [03. Основы языка](book/03-language-basics.md) | S-выражения, функции, списки, управление потоком | синтаксис, `def`, `lambda` |
| [04. Система типов](book/04-type-system.md) | Постепенная типизация, линейные типы тензоров | type hints, mypy |
| [05. Модель памяти](book/05-memory-model.md) | HibLin: scratch/stable, пулы, Scope, `defvar` | GC, `numpy` views |
| [06. Тензоры](book/06-tensors.md) | N-мерные массивы, F16/F32, broadcasting, matmul | NumPy / torch |
| [07. Автоградиент](book/07-autograd.md) | Лента градиентов, `!`-операции, SGD/Adam | `torch.autograd` |
| [08. Глубокое обучение](book/08-deep-learning.md) | Слои, `linear`, `sequential` (модуль DL) | `torch.nn` |
| [09. Классический ML](book/09-classic-ml.md) | KNN, деревья, ансамбли, метрики, CV, grid search | scikit-learn |
| [10. HLO-компиляция](book/10-hlo-compilation.md) | AOT `llc → g++ → dlopen`, fusion, `defuse`/`defuse!` | `torch.compile`, XLA |
| [11. GPU](book/11-gpu.md) | Устройства, статус CUDA-бэкенда | `.to('cuda')` |
| [12. Макросы](book/12-macros.md) | Код-как-данные, backquote, compiler macros | декораторы, метаклассы |
| [13. ООП](book/13-oop.md) | defstruct, defclass, дженерики (CLOS) | `class`, `@dataclass` |
| [14. Символьное программирование](book/14-symbolic-programming.md) | match, unify, rule engine, amb, `NS-IF`/`NS-GRAD!` | `match-case`, Prolog |
| [15. Стандартная библиотека](book/15-standard-library.md) | IO, Regex, Audio, Datetime, Errors, Visual | `re`, `open`, `datetime` |
| [16. Пакеты и FFI](book/16-packages-and-ffi.md) | qvalent, pkg-install, FFI: dlopen+FFI-CALL/IMPORT | `pip`, `ctypes` |
| [17. Разбор примеров](book/17-full-examples.md) | Линейная регрессия и XOR-MLP с Python-эквивалентами | end-to-end туториалы |
| [18. Шпаргалка](book/18-python-cheatsheet.md) | QLISP ↔ Python/NumPy/PyTorch/sklearn | справочник |
| [19. Устройство компилятора](book/19-internals-overview.md) | Reader, интерпретатор, HibLin, codegen, LSP, NS-router | CPython internals |

## Быстрый старт

```bash
# Скачать бинарник (Linux x86_64, релиз v2.2.x)
curl -L https://github.com/FQingLars/QLISP/releases/latest/download/qlisp-linux-x86_64 -o qlisp
chmod +x qlisp && mv qlisp ~/.local/bin/

# Windows: qlisp-windows-x86_64.zip со страницы релизов (qlisp.exe, qlispc.exe)

# Компилятор (AOT): qlispc — см. гл. 2

# REPL
qlisp
```

```lisp
qlisp> (+ 1 2 3 4 5)
=> 15
qlisp> (defun square (x: Float) -> Float (* x x))
=> square
qlisp> (square 4.0)
=> 16.0
qlisp> (import ML)
```

## Для кого эта книга

- **ML-инженеры на Python**, которым нужен компилируемый язык без GIL, без GC-пауз и с нативной производительностью.
- **Лисперы**, которым нужен современный инструментарий ML: тензоры, автоград, HLO-AOT, нейросимвольный движок.
- **Исследователи нейросимвольного AI**: QLISP уникально сочетает гомоиконный символьный движок (правила, унификация, паттерн-матчинг, `NS-IF`/`NS-GRAD!` с guilt detector) с нейросетевым бэктендом (автоград, HLO, GPU).

## Статус книги

Книга сопровождает **QLISP v2.2.2** — стабилизированную нейросимвольную маршрутизацию (per-sample guilt, OR-aware resolve_correct, HLO composite/broadcast/identity), миграцию памяти HibLin (Scope stable/scratch, TensorBufferPool + ConsCellPool + StableMemory), dtype F16 (`_Float16`/F16C kernels). Примеры кода проверены на интерпретаторе QLISP v2.2.2 (тесты в `test/phaseN_test.qlsp` и `tests/*.qlsp`).

Известные ограничения реализации (см. `TEMP_ISSUES.md` и `TODO.md` в исходниках):

- CUDA-бэкенд (`src/gpu/kernels.cu`) присутствует, но **не тестирован на GPU**.
- HLO-AOT делается через `popen("llc ...")` + `dlopen`, не через OrcJIT (OrcJIT крашился в QLISP).
- Поля `Tensor::is_stable` и `Tensor::size_bytes` пока не реализованы.
- TensorBufferPool внутри использует power-of-2 buckets (а не exact-size free list) — фактическая реализация отличается от идеала HibLin, описанного в `ARCHITECTURE.md`.

## Лицензия

QLISP и эта книга распространяются под лицензией MIT.