# Глава 19. Устройство компилятора

Краткая экскурсия во внутренности QLISP v2.2.2. Полное описание — в `ARCHITECTURE.md` исходного репозитория.

## 19.1 Пайплайн

```text
Исходник → CharStream → Readtable (reader) → SExpr-дерево в ConsCellPool (гл. 5)
  → MacroExpander (backquote, defmacro, compiler-macros, MACROLET)
  → Interpreter::eval (special forms, применение функций, примитивы)
    → Tensor-операции → GradientTape (запись autograd)
    → GradientTape::backward (reverse-topo AD, blocked nodes для NS-IF)
  → HloGraph (START-TRACE … HLO-COMPILE)
    → HloCodeGen → LLVM IR → llc → .o → g++ -shared → .so → dlopen (AOT, не OrcJIT)
  → NS-router (NS-IF/NS-GRAD! → per-sample routing + guilt detector)
                     ↓
                LSP Server ←→ Editor
```

## 19.2 Структура исходников

```
src/
├── core/          Scope (HibLin), SExpr, Symbol, ConsCellPool, TensorBufferPool, StableMemory
├── reader/        CharStream, Readtable
├── eval/          Interpreter (80+ примитивов), специальные формы
├── types/         Type, TypeChecker (постепенная типизация)
├── macros/        MacroExpander, backquote
├── mop/           ClassObj, ClassRegistry, Instance (CLOS)
├── tensor/        Tensor, GradientTape (F16/F32/F64/I32/I64/U8)
├── gpu/           Device (CPU активен; CUDA/ROCm в плане)
├── hlo/           HloGraph, HloNode, HloOp, оптимизации, fingerprint
├── codegen/       CodeGen (top-level AOT), HloCodeGen (HLO→native AOT)
├── symbolic/      S1-S8 (match, unify, plist, defrule, amb, compiler-macros, defhloop)
├── ns/            RouteHop, Router, GuiltDetector (S9)
├── data/          foundation, qserde (.qsrd-сериализация)
├── ffi/           FFI-LOAD/CALL/IMPORT (dlopen + dlsym)
├── visual/        Dashboard (HTTP, TensorBoard-подобный)
├── runtime/       qlisp.c, qlisp_tensor.cpp (C/C++-рантайм для AOT)
├── qvalent/       qvalent CLI (пакетный менеджер)
├── lsp/           server.cpp (только LSP, без LLVM)
├── stdlib/        *.qlsp → embedded_modules[] (через xxd -i)
└── main.cpp       REPL/интерпретатор/AOT-компилятор
```

## 19.3 Reader

Символьная диспетчеризация (`src/reader/readtable.cpp`):

| Символ | Действие |
|---|---|
| `(` | разбор списка (правильного или dotted) |
| `)` | error |
| `'` | разворачивается в `(QUOTE ...)` |
| `` ` `` / `,` / `,@` | `(BACKQUOTE ...)`, `(COMMA ...)`, `(COMMA-AT ...)` |
| `#` | dispatch: `#'` → FUNCTION, `#(` → вектор, `#\` → литера |
| `"` | строка с escape-последовательностями |
| `;` | комментарий до конца строки |

Токены: символы интернируются в верхнем регистре через `SymbolTable::instance()` (регистронезависимость); сначала пробуется парсинг числа (int64 → double), иначе — символ; `nil` → синглтон nil, `t` → символ T.

> 🐍 Аналог — фаза tokenizer/parser в CPython, но результат — не `ast.AST`, а cons-ячейки: код и данные одного представления.

## 19.4 S-выражения

Тегированное объединение (`SExpr`, `src/core/sexpr.hpp`):

```cpp
class SExpr {
    Tag tag_;       // NIL, CONS, SYMBOL, FIXNUM, FLONUM, STRING,
                    // PRIMITIVE, CLOSURE, INSTANCE, TENSOR, HLOPROG
    union {         // tagged payload
        struct { SExpr *car_, *cdr_; } cons_;
        const Symbol *sym_;
        int64_t fixnum_;
        double flonum_;
        struct { const char *data_; size_t len_; } string_;
        Tensor *tensor_;
        struct { SExpr *params_, *body_, *env_; } closure_;
        void *compiled_fn_;  // HLOPROG (dlopen'd)
    };
};
```

Все cons-ячейки выделяются в `ConsCellPool`; объекты — в `Scope` с деструктор-трекингом.

Замыкание: параметры + тело + захваченное окружение. Глобальные `defun`/`defmacro`/`defrule` сохраняются в `root_arena_` (stable) — код-как-данные живёт всю программу (гомоиконность рантайма).

## 19.5 Окружение и вычисление

```cpp
struct EnvFrame {
    const EnvFrame *parent;   // лексический родитель (nullptr для global)
    SExpr *bindings;          // alist: ((sym . val) (sym2 . val2) ...)
};
```

Реализация: `EnvFrame` использует C++ `std::map` для bindings (см. `TEMP_ISSUES.md` #2 — планируется SExpr-alist в Phase 12).

Вычисление: атом → сам; символ → поиск переменной; cons → special form или вызов функции. Замыкания захватывают цепочку окружений в плоский alist (`do_lambda` в `interpreter.cpp:572`).

## 19.6 Special forms

Из `sym_*` в `src/eval/interpreter.cpp:53-105`:

`QUOTE, IF, PROGN, LAMBDA(\), LET, LET*, DEFUN, DEFUSE, DEFUSE!, DEFMACRO, MACROLET, DEFVAR, SETQ, COND, WHILE, AND, OR, BACKQUOTE, IMPORT, THE, GENSYM, MACROEXPAND, MACROEXPAND-1, FUNCALL, APPLY, DEFCLASS, MAKE-INSTANCE, SLOT-VALUE, DEFSTRUCT, DEFGENERIC, DEFMETHOD, MATCH, UNIFY, SUBST, AMB-LET, REQUIRE, AMB-COLLECT, ASSERT-FACT, RETRACT-FACT, DEFRULE, QUERY, RUN-RULES, DEFINE-COMPILER-MACRO, DEFHLOOP, NS-IF, NS-GRAD!, TENSOR` — и **80+ примитивов** (арифметика, тензоры, autograd, HLO, FFI, IO, data).

## 19.7 Тензорная подсистема

```text
Tensor: dtype (F16/F32/F64/I32/I64/U8), shape, strides, numel,
        data (TensorBufferPool — exact-size, 0% перерасхода),
        requires_grad, grad, grad_node (лента), m/v/step (Adam)
```

- Поэлементные операции — SIMD: AVX2 (8×f32) на x86_64, NEON (4×f32) на ARM.
- `MATMUL` — `cblas_sgemm` (OpenBLAS).
- `reshape`/`transpose` — глубокие копии (views нет, гл. 4).
- `TensorBufferPool`: exact-size free list — буфер выделяется ровно под размер тензора, при освобождении возвращается в список своего размера; лимит пула — 1 ГБ (настраивается).
- F16: `launch_*_half` через `_Float16` (F16C на x86_64); matmul аккумулирует во float.

## 19.8 Autograd

Узел ленты (`src/tensor/gradient_tape.hpp`):

```cpp
struct GradNode { Tensor *result;
                  std::vector<Tensor*> inputs;
                  std::function<void(Scope&)> backward_fn; };

class GradientTape {  // thread_local singleton
    static GradientTape& instance();
    void push(GradNode* node);
    void backward(Tensor* loss, Scope& arena, blocked_set_t* blocked = nullptr);
    void clear();
};
```

`backward` идёт по dataflow-графу в reverse-topo (потребители раньше производителей) через `result → producer`-карту, накапливая `+=` градиенты в `inputs`. `GradNode::inputs` хранит `Tensor*` (не SExpr) — autograd не завязан на интерпретатор.

Нейросимвольные `NS-IF`/`NS-GRAD!` (см. гл. 14) добавляют «вину»: STE-градиент для роутера, блокировка ошибочных ветвей через `blocked_set_t`.

## 19.9 HLO-пайплайн

Граф из узлов `HloOp` (см. `src/hlo/graph.hpp`): `PARAMETER, CONSTANT, DOT, ADD, MUL, RELU, FUSION, TUPLE, OUTPUT, LOOKUP, ARGMAX, SOFTMAX, COMPOSITE, CODEGEN, WEIGHTED_LOOKUP`.

Оптимизации: CSE, DCE, fusion поэлементных цепочек, broadcast-shape, COMPOSITE-expansion. Кодоген (`src/codegen/hlo_codegen.cpp`): C-ABI-функция `hlo_fn(params, output)` → LLVM IR → `popen("llc … && g++ -shared …")` → `dlopen` (AOT, **не OrcJIT**). Кэш: `/tmp/qlisp_hlo_cache/` с versioned fingerprint. Fallback: если AOT не удался — интерпретация клонированного графа (`HloGraph::clone()`).

Range_fn генерируется только для полностью fused графов (entry=FUSION, или RELU/ADD/MUL с PARAMETER-входами).

## 19.10 Neuro-Symbolic Router

`src/ns/route.{hpp,cpp}`:

- `RouteHop` — одна символьная развилка: `decision_logits`, `batch`, `taken[]`, `correct[]`, `branch_covers[]`, диапазон tape-узлов.
- `RouteHop::resolve_correct(label, taken)` — case-insensitive; OR-aware: taken-ветвь первая; если она покрывает label → маршрут верный (виновен лист).
- `Router` — упорядоченные хопы от корня к листу с depth-трекингом.
- `GuiltDetector::detect(router, labels)` → `GuiltVerdict { kind, guilty_hop, per_sample_hop, per_sample_kind }`.
- `NS-IF` (`do_ns_if`) выбирает ветвь per-sample по argmax, записывает хоп, исполняет ветвь первого образца (для чистого батча; для смешанного — маршрут/guilt per-sample, но исполнение одно).
- `NS-GRAD!` (`do_ns_grad`) блокирует tape-узлы виновных ветвей, добавляет softmax-weighted STE в `decision_logits->grad`.

## 19.11 LSP

`qlisp-lsp` (`src/lsp/server.cpp`, отдельная цель CMake без LLVM) — Language Server без внешних зависимостей (JSON-парсер встроен): 150+ автодополнений с hover-документацией, разбор `defun/defvar/defmacro/defclass/defstruct/defhloop/defrule`, подсказки параметров, goto-definition.

## 19.12 Сборка и инструменты

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

Продукты сборки (из `CMakeLists.txt`):

- `qlisp` — REPL/интерпретатор (с LLVM, OpenBLAS, OpenMP).
- `qlispc` — AOT-компилятор (те же исходники).
- `qlisp-lsp` — LSP-сервер (только `src/lsp/server.cpp`, без LLVM).
- `qvalent` — пакетный менеджер (только `src/qvalent/main.cpp`).
- `libqlisp_runtime.a` — C-рантайм для AOT-линковки.
- `libqlisp_tensor_runtime.a` — тензорный рантайм (C++: AVX2/FMA + cblas_sgemm).

Модули stdlib вшиваются в бинарник на этапе сборки (`cmake/stdlib_embed.cmake`, через `xxd -i`) — поэтому дистрибутив QLISP — один файл ~2 МБ. Команда зависит от `xxd`, что может быть проблемой для MSYS2/MinGW (см. `TODO.md` §3).

## 19.13 Ключевые константы

| Параметр | Значение |
|---|---|
| Блок scratch-Scope | 16 МБ (`DEFAULT_BLOCK_SIZE`) |
| ConsCellPool | предвыделенный, `BLOCK_CELLS` ячеек за блок |
| TensorBufferPool | exact-size, лимит 1 ГБ |
| SIMD (AVX2/FMA) | 8×f32 = 256 бит |
| SIMD (NEON) | 4×f32 = 128 бит |
| Кэш HLO | `/tmp/qlisp_hlo_cache/` |
| Модули stdlib вшито | 9 (core ml io regex audio datetime pkg errors visual) |
| Доп. модуль dl | вшит наравне с остальными (`stdlib/dl.qlsp`, 24 строки) |
| Автодополнения LSP | 150+ |
| LLVM (минимум) | 22+ |
| GCC/Clang (минимум) | GCC 13+ / Clang 16+ (C++20) |
| qvalent локальный каталог | `./qlisp-packages/<name>/` |
| qvalent кэш | `$HOME/.cache/qlisp/<deployer>/<repo>/` |

## 19.14 Известные ограничения (см. TEMP_ISSUES.md и TODO.md)

- **CUDA** (`src/gpu/kernels.cu`) не зарегистрирован и не тестирован.
- **HLO-AOT** через `popen` (не OrcJIT) — первый вызов может быть медленным.
- **EnvFrame** использует C++ `std::map` — будет переписан на SExpr-alist в Phase 12.
- **`Tensor::is_stable`** и **`Tensor::size_bytes`** пока не реализованы.
- **`graph-node`/`graph-rewrite`** (публичное API для переписывания HloGraph из языка) — не реализованы.
- **ROUTE-узел** в HLO — не реализован; `do_ns_if` сейчас разрывает ленту на точке выбора, а не строит ROUTE.
- **Кросс-компиляция** — `QLISP_TARGET_TRIPLE/QLISP_TARGET_CPU/QLISP_LLC/QLISP_CXX` env vars.
- **CI** — нет GitHub Actions, нет CTest; все тесты — `.qlsp`-скрипты в `test/` (standalone) и `tests/` (`deftest`/`run-tests`).