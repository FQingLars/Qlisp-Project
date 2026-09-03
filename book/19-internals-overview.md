# Глава 19. Устройство компилятора

Краткая экскурсия во внутренности QLISP v2.3.0. Полное описание — в `ARCHITECTURE.md` исходного репозитория.

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

### Контракт tape v2.3.0 (исправления TBC #26)

С v2.3.0 контракт ужесточён и формализован:

1. **Tape-узлы принадлежат tape.** Аллоцируются в heap, удаляются в `clear()`. До v2.3.0 жили в scratch-блоках (UB: `heap-use-after-free` под ASan при `defvar` промежуточного тензора и `grad!` в следующей форме).

2. **`defvar` промоутит промежуточный тензор.** `defvar`/`setq` клонирует тензор в stable-scope и перенаправляет ребро `result` tape-ноды на stable-копию. Градиент течёт в тензор, который видит пользователь.

3. **`reset_scratch` «запечатывает» незакрытые графы.** Если форма верхнего уровня завершилась без `GRAD!`, и какие-то tape-узлы ссылаются на scratch-тензоры, эти тензоры один раз клонируются в stable, и рёбра переписываются. Обучающие циклы ничего не платят (tape пуст после каждого `GRAD!`).

4. **Биндинги не алиасят литералы.** `let`/`bind_and_eval` копируют скалярные значения (fixnum/flonum/symbol) в свежие ячейки — `setq` больше не мутирует исходные литералы в коде.

5. **Adam-состояние живёт на параметре.** `m`/`v`/`step` аллоцируются в stable-scope на самом параметре (исправление `f61426b`); раньше умирали с итерацией, momentum не работал.

6. **HLO/NS-прокси стали строковыми хендлами.** Биндинги копируют скаляры, и fixnum-прокси терял идентичность (ключ `trace_nodes_`). Теперь — строковые хендлы, детерминированно.

## 19.9 HLO-пайплайн

Граф из узлов `HloOp` (см. `src/hlo/graph.hpp`): `PARAMETER, CONSTANT, DOT, ADD, MUL, RELU, FUSION, TUPLE, OUTPUT, LOOKUP, ARGMAX, SOFTMAX, COMPOSITE, CODEGEN, WEIGHTED_LOOKUP`.

Оптимизации: CSE, DCE, fusion поэлементных цепочек, broadcast-shape, COMPOSITE-expansion. Кодоген (`src/codegen/hlo_codegen.cpp`): C-ABI-функция `hlo_fn(params, output)` → LLVM IR → object → `dlopen` (AOT, **не OrcJIT**).

С **v2.3.0** путь **полностью in-process** (коммит `ef4994e`): `TargetMachine::emit` → `lld::elf::link` → `dlopen`. Никакого `popen`/`system`. Детект наличия LLD через CMake (`QLISP_HAS_LLD`). Если LLD не подключена при сборке — fallback на shell-линковку `popen("llc … && g++ -shared …")`. Линкеры Debian-style требуют явный `-lz -lzstd`; CI обновлён: `lld-22 liblld-22-dev`.

Кэш: `/tmp/qlisp_hlo_cache/` с versioned fingerprint. Fallback: если AOT не удался — интерпретация клонированного графа (`HloGraph::clone()`).

Range_fn генерируется только для полностью fused графов (entry=FUSION, или RELU/ADD/MUL с PARAMETER-входами).

### Pin LLVM по `llvm-config` (v2.3.0+, коммит `1bf13b4`)

`find_package(LLVM)` в CMake теперь запинен к установке, на которую указывает `llvm-config` из PATH. Раньше был возможен ABI-микс (заголовочные файлы из одного LLVM, библиотеки — из другого) — приводил к крашам и неопределённому поведению. Теперь:

- Сборка выбирает `llvm-config` из PATH (или `PATH=/opt/rocm/lib/llvm/bin:$PATH cmake ..`).
- Include и lib берутся от одного LLVM (`llvm-config --includedir` + `llvm-config --libfiles`).
- Это устраняет ABI-конфликты вроде `undefined reference to operator new(unsigned long)` при кросс-линковке host-LLVM с MinGW.

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
# Native Linux
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Если LLVM в нестандартном месте (например, ROCm)
PATH=/opt/rocm/lib/llvm/bin:$PATH cmake .. -DCMAKE_BUILD_TYPE=Release \
  && PATH=/opt/rocm/lib/llvm/bin:$PATH make -j$(nproc)

# Cross-compile qvalent/qlisp-lsp под Windows (qlisp/qlispc требуют нативной сборки MSYS2)
PATH=/opt/rocm/lib/llvm/bin:$PATH cmake -B build-windows \
  -DCMAKE_TOOLCHAIN_FILE=cmake/x86_64-w64-mingw32.cmake \
  -DCMAKE_BUILD_TYPE=Release
```

Продукты сборки (из `CMakeLists.txt`):

- `qlisp` — REPL/интерпретатор (с LLVM, OpenBLAS, OpenMP, in-process LLD).
- `qlispc` — AOT-компилятор (те же исходники).
- `qlisp-lsp` — LSP-сервер (только `src/lsp/server.cpp`, без LLVM).
- `qvalent` — пакетный менеджер (только `src/qvalent/main.cpp`).
- `libqlisp_runtime.a` — C-рантайм для AOT-линковки.
- `libqlisp_tensor_runtime.a` — тензорный рантайм (C++: AVX2/FMA + cblas_sgemm).

Модули stdlib вшиваются в бинарник на этапе сборки (`cmake/stdlib_embed.cmake`, через `cmake/gen_stdlib_header.cmake` — без `xxd` для переносимости на MSYS2/MinGW) — поэтому дистрибутив QLISP — один файл ~2 МБ.

### Windows-сборка

Локальная кросс-компиляция `qlisp`/`qlispc` из Linux **невозможна** для v2.3.0: host-LLVM (например, rocm `libLLVM*.a`) собран с SysV ABI, а MinGW — с Win64 ABI; объектные файлы принципиально несовместимы (`undefined reference to operator new(unsigned long)`, `multiple definition std::_Sp_make_shared_tag` и т.п.). Кросс-путь в CMake оставлен **только для `qlisp-lsp` и `qvalent`** (без LLVM, работает; добавлены шимы заголовков `uid_t/gid_t/nlink_t`).

Корректные пути для Windows-бинарника:

1. **CI** (`.github/workflows/build.yml`, job `windows`): MSYS2 MinGW64 + LLVM/OpenBLAS нативно под Win64 ABI. Публикация релиза автоматическая при пуше тега `v2.3.0` (`git tag v2.3.0 && git push origin main v2.3.0`).
2. **MSYS2 локально**: установить `mingw-w64-x86_64-llvm`/`mingw-w64-x86_64-clang`/`mingw-w64-x86_64-openblas`, затем `cmake -G "MinGW Makefiles" .. && make -j$(nproc)`.

## 19.13 Ключевые константы

| Параметр | Значение |
|---|---|
| Блок scratch-Scope | 16 МБ (`DEFAULT_BLOCK_SIZE`) |
| ConsCellPool | предвыделенный, `BLOCK_CELLS` ячеек за блок |
| TensorBufferPool | exact-size, лимит 1 ГБ |
| SIMD (AVX2/FMA) | 8×f32 = 256 бит |
| SIMD (NEON) | 4×f32 = 128 бит |
| Кэш HLO | `/tmp/qlisp_hlo_cache/` (in-process LLD, v2.3.0+) |
| Модули stdlib вшито | 11 (core string dl ml io regex audio datetime pkg errors visual) |
| `dl` v2 | слои-как-данные, ~100 строк, `linear`/`net-forward`/`train-step`/`train-epochs` |
| Автодополнения LSP | 150+ |
| LLVM (минимум) | 22+ (с `lld` для in-process AOT) |
| GCC/Clang (минимум) | GCC 13+ / Clang 16+ (C++20) |
| qvalent локальный каталог | `./qlisp-packages/<name>/` |
| qvalent кэш | `$HOME/.cache/qlisp/<deployer>/<repo>/` |
| QSRD v2 magic | `"QSRD"` (u16 ver) |
| Тесты | 48/50 (Release, Linux); ASan чисто на tape-тяжёлых |

## 19.14 Известные ограничения (см. TEMP_ISSUES.md и TODO.md)

- **CUDA** (`src/gpu/kernels.cu`) не зарегистрирован и не тестирован.
- **HLO-AOT** через `popen` (не OrcJIT) — fallback, если LLD не подключена при сборке. С v2.3.0 по умолчанию **in-process LLD** (`ef4994e`).
- **EnvFrame** использует C++ `std::map` — будет переписан на SExpr-alist в Phase 12.
- **`Tensor::is_stable`** и **`Tensor::size_bytes`** пока не реализованы.
- **`graph-node`/`graph-rewrite`** (публичное API для переписывания HloGraph из языка) — не реализованы.
- **ROUTE-узел** в HLO — не реализован; `do_ns_if` сейчас разрывает ленту на точке выбора, а не строит ROUTE.
- **HLO fallback-граф (без `HLO-COMPILE`)** — держит константы формы трассировки, кросс-форменное использование небезопасно (`TEMP_ISSUES` #23).
- **Кросс-компиляция `qlisp`/`qlispc` под Windows из Linux** — невозможна (ABI-mix SysV/Win64). Используйте CI или MSYS2.
- **CI** — GitHub Actions (`build.yml`) на Linux и Windows; тесты — `.qlsp`-скрипты в `test/` (standalone) и `tests/` (`deftest`/`run-tests`).
- **Два детерминированных фейла** (вне скоупа 2.3.0): `tests/ns_router.qlsp` → `ns-nested-breed-guilty`; `tests/symbolic_neuro.qlsp` → `neurosym-rule-check`.