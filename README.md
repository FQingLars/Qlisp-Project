# 📖 QLISP Book

**Ориентировочная книга по языку программирования QLISP** — компилируемому Лиспу для машинного обучения и AI.

QLISP — это диалект Лиспа, который компилируется в нативный машинный код через LLVM, со встроенной тензорной системой (F16-поддержка с v2.2.0), автоградиентом, HLO-компиляцией с AOT-инференсом и нейросимвольным движком (rule engine, унификация, `NS-IF`/`NS-GRAD!` с v2.2.0+) — всё в стандартной библиотеке, без единого `pip install`. v2.3.0 приносит гомоиконность (`EVAL`/`READ-FROM-STRING`/`FUNCTION-BODY`/`SYMBOL-NAME`), QSRD v2 для бинарной сериализации, модуль `string`, частичный импорт `IMPORT-FROM`, переписанный `dl` со слоями-как-данные, переживающий границы форм gradient tape и in-process LLD для HLO-компиляции.

Книга написана для **инженеров с Python-бэкграундом**: каждая глава содержит аналогии с Python, NumPy, PyTorch и scikit-learn.

## Оглавление

| Глава | Тема | Python-аналог |
|-------|------|---------------|
| [01. Введение](book/01-introduction.md) | Зачем нужен QLISP, философия, пайплайн | «Почему не Python?» |
| [02. Начало работы](book/02-getting-started.md) | Установка, REPL, первая программа, компиляция | `python`, `pip` |
| [03. Основы языка](book/03-language-basics.md) | S-выражения, функции, гомоиконность (`EVAL`, `FUNCTION-BODY`) | синтаксис, `def`, `lambda`, `ast` |
| [04. Система типов](book/04-type-system.md) | Постепенная типизация, линейные типы тензоров | type hints, mypy |
| [05. Модель памяти](book/05-memory-model.md) | HibLin: scratch/stable, пулы, Scope, `defvar`, `reset_scratch` | GC, `numpy` views |
| [06. Тензоры](book/06-tensors.md) | N-мерные массивы, F16/F32, broadcasting, matmul | NumPy / torch |
| [07. Автоградиент](book/07-autograd.md) | Лента градиентов, `!`-операции, SGD/Adam, контракт tape между формами | `torch.autograd` |
| [08. Глубокое обучение](book/08-deep-learning.md) | Модуль DL v2: слои-как-данные, `net-forward`/`train-step`/`train-epochs` | `torch.nn` |
| [09. Классический ML](book/09-classic-ml.md) | KNN, деревья, ансамбли, метрики, CV, grid search | scikit-learn |
| [10. HLO-компиляция](book/10-hlo-compilation.md) | In-process AOT: LLVM emit → object → lldELF → dlopen | `torch.compile`, XLA |
| [11. GPU](book/11-gpu.md) | Устройства, статус CUDA-бэкенда | `.to('cuda')` |
| [12. Макросы](book/12-macros.md) | Код-как-данные, backquote, compiler macros, `EVAL`/`READ-FROM-STRING` | декораторы, метаклассы |
| [13. ООП](book/13-oop.md) | defstruct, defclass, дженерики (CLOS) | `class`, `@dataclass` |
| [14. Символьное программирование](book/14-symbolic-programming.md) | match, unify, rule engine, amb, `NS-IF`/`NS-GRAD!` | `match-case`, Prolog |
| [15. Стандартная библиотека](book/15-standard-library.md) | IO, Regex, Audio, Datetime, Errors, Visual, **String** | `re`, `open`, `datetime` |
| [16. Пакеты и FFI](book/16-packages-and-ffi.md) | qvalent, pkg-install, FFI: dlopen+FFI-CALL/IMPORT | `pip`, `ctypes` |
| [17. Разбор примеров](book/17-full-examples.md) | Линейная регрессия и XOR-MLP с Python-эквивалентами | end-to-end туториалы |
| [18. Шпаргалка](book/18-python-cheatsheet.md) | QLISP ↔ Python/NumPy/PyTorch/sklearn | справочник |
| [19. Устройство компилятора](book/19-internals-overview.md) | Reader, интерпретатор, HibLin, codegen, LSP, NS-router | CPython internals |

## Быстрый старт

```bash
# Скачать бинарник (Linux x86_64, релиз v2.3.0)
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

Книга сопровождает **QLISP v2.3.0** — стабилизированную нейросимвольную маршрутизацию (per-sample guilt, OR-aware resolve_correct, HLO composite/broadcast/identity), миграцию памяти HibLin (Scope stable/scratch, TensorBufferPool + ConsCellPool + StableMemory), dtype F16 (`_Float16`/F16C kernels), гомоиконный рантайм (`EVAL`/`READ-FROM-STRING`/`FUNCTION-BODY`/`SYMBOL-NAME`), QSRD v2 (бинарный формат), 11 стандартных модулей (включая `string`), `IMPORT-FROM` для частичного импорта, переписанный модуль `dl` (слои как данные, переживают границы форм), gradient tape с корректным `reset_scratch`/`defvar_autograd` и in-process LLD для AOT HLO-компиляции (без `popen`).

Известные ограничения реализации (см. соответствующие разделы в книге):

- CUDA-бэкенд (`src/gpu/kernels.cu`) присутствует, но **не тестирован на GPU**.
- Windows: локальная кросс-компиляция `qlisp`/`qlispc` из Linux невозможна (SysV-ABI в host-LLVM несовместим с Win64 ABI). Используйте CI-релиз или MSYS2-сборку.
- HLO-fallback (без `HLO-COMPILE`) держит константы формы трассировки — кросс-форменное использование такого HLOPROG небезопасно.
- `ns-nested-breed-guilty` и `neurosym-rule-check` — два детерминированных фейла в test-suite, существовали до 2.3.0 и вне скоупа релиза.

## Лицензия

QLISP и эта книга распространяются под лицензией MIT.