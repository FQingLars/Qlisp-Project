# Глава 19. Устройство компилятора

Краткая экскурсия во внутренности QLISP (для полного описания см. `ARCHITECTURE.md` исходного репозитория).

## 19.1 Пайплайн

```text
Исходник → CharStream → Readtable (reader) → SExpr-дерево в арене
  → MacroExpander (backquote, макросы, compiler macros)
  → Interpreter::eval (special forms, применение функций, примитивы)
    → Tensor-операции → GradientTape (запись autograd)
    → GradientTape::backward (reverse-mode AD)
  → HloGraph (режим трассировки)
    → HloCodeGen → LLVM IR → llc+g++ → .so → dlopen → нативное исполнение
```

## 19.2 Reader

Символьная диспетчеризация:

| Символ | Действие |
|---|---|
| `(` | разбор списка (правильного или dotted) |
| `'` | разворачивается в `(QUOTE ...)` |
| `` ` `` / `,` / `,@` | `(BACKQUOTE ...)`, `(COMMA ...)`, `(COMMA-AT ...)` |
| `#` | dispatch: `#'` → FUNCTION, `#(` → вектор, `#\` → литера |
| `"` | строка с escape-последовательностями |
| `;` | комментарий до конца строки |

Токены: символы интернируются в верхнем регистре (регистронезависимость); сначала пробуется парсинг числа (int64 → double), иначе — символ; `nil` → синглтон nil, `t` → символ T.

> 🐍 Аналог — фаза tokenizer/parser в CPython, но результат — не `ast.AST`, а cons-ячейки: код и данные одного представления.

## 19.3 S-выражения

Тегированное объединение (`Tag`): `NIL, CONS, SYMBOL, FIXNUM, FLONUM, STRING, PRIMITIVE, CLOSURE, INSTANCE, TENSOR, HLOPROG`. Всё выделяется в аренах (гл. 5), никаких деструкторов по одному.

Замыкание: параметры + тело + захваченное окружение. Глобальные `defun`/`defmacro` клонируются в immortal — код-как-данные живёт вечно (гомоиконность рантайма).

## 19.4 Окружение и вычисление

```cpp
struct EnvFrame {
    const EnvFrame *parent;   // лексический родитель
    SExpr *bindings;          // alist: ((sym . val) ...)
};
```

Вычисление: атом → сам; символ → поиск переменной; cons → special form или вызов функции. Замыкания захватывают цепочку окружений в плоский alist.

## 19.5 Special forms

`QUOTE, IF, PROGN, LAMBDA(\), LET, LET*, DEFUN, DEFUSE, DEFUSE!, DEFMACRO, DEFVAR, SETQ, COND, WHILE, AND, OR, BACKQUOTE, IMPORT, THE, GENSYM, MACROEXPAND, FUNCALL, DEFCLASS, MAKE-INSTANCE, SLOT-VALUE, DEFSTRUCT, DEFGENERIC, DEFMETHOD, MATCH, UNIFY, SUBST, AMB-LET, AMB-COLLECT, ASSERT-FACT, RETRACT-FACT, DEFRULE, QUERY, RUN-RULES, DEFCOMPILER-MACRO, DEFHLOOP, NS-IF, NS-GRAD!, LOAD, REQUIRE, TENSOR` — и 70+ примитивов (арифметика, тензоры, autograd, HLO, FFI, IO, data).

## 19.6 Тензорная подсистема

```text
Tensor: dtype (F32/F64/I32/I64/U8), shape, strides, numel,
        data (device-пул), device (CPU/CUDA/ROCm),
        requires_grad, grad, grad_node (лента), m/v/step (Adam)
```

- Поэлементные операции — SIMD: AVX2 (8×f32) на x86_64, NEON (4×f32) на ARM.
- `MATMUL` — `cblas_sgemm` (OpenBLAS).
- `reshape`/`transpose` — глубокие копии (views нет, гл. 4).
- TensorBufferPool: бакеты по размеру (степени двойки от 256 байт), ≤16 записей на бакет, TTL 10 секунд.

## 19.7 Autograd

Узел ленты: результат + входы + backward-функция. `GRAD!` идёт по ленте в обратном порядке и разматывает цепное правило; после прохода лента очищается. Нейросимвольные `NS-IF`/`NS-GRAD!` добавляют узлам «вину» (гл. 14): STE-градиент для роутера, блокировка ошибочных ветвей.

## 19.8 HLO-пайплайн

Граф из узлов `HloOp: PARAMETER, CONSTANT, DOT, ADD, MUL, RELU, FUSION, SOFTMAX, ARGMAX, LOOKUP, WEIGHTED_LOOKUP, COMPOSITE, CODEGEN`. Оптимизации: CSE, DCE, fusion поэлементных цепочек. Кодоген: C-ABI-функция `hlo_fn(params, output)` → LLVM IR → `llc` → `g++ -shared` → `dlopen`. Кэш: `/tmp/qlisp_hlo_cache` с версионированным отпечатком. Fallback: если AOT не удался — интерпретация клонированного графа.

## 19.9 LSP

`qlisp-lsp` — Language Server без внешних зависимостей (JSON-парсер встроен): 150+ автодополнений с hover-документацией, разбор `defun/defvar/defmacro/defclass/defstruct/defhloop`, подсказки параметров, goto-definition.

## 19.10 Сборка и инструменты

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

Продукты сборки: `qlisp` (REPL/интерпретатор), `qlispc` (компилятор), `qlisp-lsp`, `qvalent` (пакетный менеджер), `libqlisp_runtime.a` (C-рантайм для AOT-линковки).

Модули stdlib вшиваются в бинарник на этапе сборки (`stdlib_embed.h`) — поэтому дистрибутив QLISP — один файл ~2 МБ.

## 19.11 Ключевые константы

| Параметр | Значение |
|---|---|
| Блок арены по умолчанию | 64 КБ |
| SIMD (AVX2) | 8×f32 = 256 бит |
| SIMD (NEON) | 4×f32 = 128 бит |
| Бакет TensorPool от | 256 байт |
| Максимум записей/бакет | 16 |
| TTL TensorPool | 10 с |
| Кэш HLO | `/tmp/qlisp_hlo_cache` |
| Модули stdlib | 10 |
| Автодополнения LSP | 150+ |
