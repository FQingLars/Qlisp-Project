# 📖 QLISP Book

**Ориентировочная книга по языку программирования QLISP** — компилируемому Лиспу для машинного обучения и AI.

QLISP — это диалект Лиспа, который компилируется в нативный машинный код через LLVM, со встроенной тензорной системой, автоградиентом, GPU-привязками и классическим ML — всё в стандартной библиотеке, без единого `pip install`.

Книга написана для **инженеров с Python-бэкграундом**: каждая глава содержит аналогии с Python, NumPy, PyTorch и scikit-learn.

## Оглавление

| Глава | Тема | Python-аналог |
|-------|------|---------------|
| [01. Введение](book/01-introduction.md) | Зачем нужен QLISP, философия | «Почему не Python?» |
| [02. Начало работы](book/02-getting-started.md) | Установка, REPL, первая программа, компиляция | `python`, `pip` |
| [03. Основы языка](book/03-language-basics.md) | S-выражения, функции, списки, управление потоком | синтаксис, `def`, `lambda` |
| [04. Система типов](book/04-type-system.md) | Постепенная типизация, линейные типы тензоров | type hints, mypy |
| [05. Модель памяти](book/05-memory-model.md) | Paper-memory: scratch/immortal, арены, `clone` | GC, `numpy` views |
| [06. Тензоры](book/06-tensors.md) | N-мерные массивы, broadcasting, матричные операции | NumPy / torch |
| [07. Автоградиент](book/07-autograd.md) | Лента градиентов, `!`-операции, оптимизаторы | `torch.autograd` |
| [08. Глубокое обучение](book/08-deep-learning.md) | Слои, CNN-операции, нормализация | `torch.nn` |
| [09. Классический ML](book/09-classic-ml.md) | KNN, деревья, ансамбли, кластеризация, метрики | scikit-learn |
| [10. HLO-компиляция](book/10-hlo-compilation.md) | Трассировка графов, fusion, `defuse` | `torch.compile`, XLA |
| [11. GPU](book/11-gpu.md) | Устройства, авто-размещение, CPU fallback | `.to('cuda')` |
| [12. Макросы](book/12-macros.md) | Код-как-данные, backquote, compiler macros | декораторы, метаклассы |
| [13. ООП](book/13-oop.md) | defstruct, defclass, дженерики | `class`, `@dataclass` |
| [14. Символьное программирование](book/14-symbolic-programming.md) | Pattern matching, унификация, rule engine, нейросимволика | `match-case`, Prolog |
| [15. Стандартная библиотека](book/15-standard-library.md) | IO, Regex, Audio, Datetime, Errors, Visual | `re`, `open`, `datetime` |
| [16. Пакеты и FFI](book/16-packages-and-ffi.md) | qvalent, pkg-install, вызов C-библиотек | `pip`, `ctypes` |
| [17. Разбор примеров](book/17-full-examples.md) | Линейная регрессия и XOR-MLP с Python-эквивалентами | end-to-end туториалы |
| [18. Шпаргалка](book/18-python-cheatsheet.md) | QLISP ↔ Python/NumPy/PyTorch/sklearn | справочник |
| [19. Устройство компилятора](book/19-internals-overview.md) | Reader, интерпретатор, кодоген, LSP | CPython internals |

## Быстрый старт

```bash
# Скачать бинарник (Linux)
curl -L <release-url>/qlisp-linux-x86_64 -o qlisp
chmod +x qlisp && mv qlisp ~/.local/bin/

# REPL
qlisp
```

```lisp
qlisp> (+ 1 2 3 4 5)
15
qlisp> (defun square (x: Float) -> Float (* x x))
qlisp> (square 4.0)
16.0
qlisp> (import ML)
```

## Для кого эта книга

- **ML-инженеры на Python**, которым нужен компилируемый язык без GIL, без GC-пауз и с нативной производительностью.
- **Лисперы**, которым нужен современный инструментарий ML: тензоры, автоград, GPU, HLO-компиляция.
- **Исследователи нейросимвольного AI**: QLISP uniquely сочетает гомоиконный символьный движок (правила, унификация, паттерн-матчинг) с нейросетевым бэктендом (автоград, HLO, GPU).

## Статус книги

Это **ориентировочная (предварительная)** редакция. Книга сопровождает QLISP v2.1.0. Примеры кода проверены на интерпретаторе QLISP.

## Лицензия

QLISP и эта книга распространяются под лицензией MIT.
