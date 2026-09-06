# Глава 19. Устройство компилятора

Краткая экскурсия во внутренности QLISP v2.3.8. Полное описание — в `ARCHITECTURE.md` дерева разработки.

## 19.1 Пайплайн

```text
Исходник → CharStream → Readtable (reader) → SExpr-дерево в ConsCellPool (гл. 5)
  → MacroExpander (backquote, defmacro, compiler-macros, MACROLET)
  → Interpreter::eval (special forms, применение функций, примитивы)
    → Tensor-операции → GradientTape (запись autograd)
    → GradientTape::backward (reverse-topo AD, blocked nodes для NS-IF)
  → HloGraph (START-TRACE … HLO-COMPILE)
    → HloCodeGen → LLVM IR → object → in-process lldELF → .so → dlopen (AOT, не OrcJIT)
    → NS-router (NS-IF/NS-GRAD! → per-sample routing + guilt detector; ROUTE-нода в HLO)
                     ↓
                LSP Server ←→ Editor
```

## 19.2 Структура исходников

```
src/
├── core/          Scope (HibLin), SExpr, Symbol, ConsCellPool, TensorBufferPool, StableMemory, simd.hpp (slab-ядра)
├── reader/        CharStream, Readtable
├── eval/          Interpreter (80+ примитивов), специальные формы
├── types/         Type, TypeChecker (постепенная типизация)
├── macros/        MacroExpander, backquote
├── mop/           ClassObj, ClassRegistry, Instance (CLOS)
├── tensor/        Tensor, GradientTape (F16/F32/F64/I32/I64/U8)
├── gpu/           Device (CPU активен; CUDA/ROCm в плане)
├── hlo/           HloGraph, HloNode, HloOp (вкл. ROUTE), оптимизации, fingerprint
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

- Поэлементные операции — SIMD: AVX2 (8×f32) на x86_64, NEON (4×f32) на ARM. С v2.3.3 — slab-ядра на raw pointers (`src/core/simd.hpp`: `sum_ps`/`max_ps`/`scale_ps`/`exp_ps` с полиномом AVX2), диспетчеризация по layout, одометр без аллокаций — sum-4096: 343ms → 4.7ms; тайловый transpose 64×64: 158ms → 4.8ms; векторизованный softmax + O(D) backward: 16.5ms → 0.40ms; memcpy-im2col conv2d: 107.8ms → 9.8ms.
- `MATMUL` — `cblas_sgemm` (OpenBLAS); с v2.3.3 — флаги `atrans`/`btrans` (транспонирования внутри GEMM), backward-ноги и шаги SGD/ADAM — на сырых указателях (ag-matmul-1024 31ms < PyTorch 34ms; train-sgd/adam-mlp-64 1.14ms).
- `reshape`/`transpose` — глубокие копии (views нет, гл. 4).
- `TensorBufferPool`: exact-size free list — буфер выделяется ровно под размер тензора, при освобождении возвращается в список своего размера; лимит пула — 1 ГБ (настраивается).
- F16: `launch_*_half` через `_Float16` (F16C на x86_64); matmul аккумулирует во float.
- `while` с v2.3.3: результат тела — в паре ping-pong scratch-скоупов (O(1) памяти на итерацию, в root продвигается один раз на выходе); утечки ловит soak-харнесс (`tests/memory_soak.qlsp` + `bench/leak_watch.py`, VmRSS по `/proc`).

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

6. **HLO/NS-прокси — строковые хендлы.** Биндинги копируют скаляры, и fixnum-прокси терял идентичность (ключ `trace_nodes_`). Теперь — строковые хендлы `"#hlo-node:<id>"`, детерминированно (уточнено в 2.3.1, #27: биндинг-копии и promote-клоны сохраняют идентичность; `HLO-COMPILE` на чужом хендле — внятная ошибка вместо сегфолта).

7. **Времена жизни градиентов (2.3.1, #28).** При входе в `backward()` на каждый нодовый тензор регистрируется guard (`Scope::track_dtor` + `Tensor::clear_grad`): при смерти scratch-региона `->grad` честно обнуляется — следующий backward не пишет в переиспользованный пул (истёкший grad — настоящий null).

8. **HLOPROG переживает promote (2.3.1, #29).** `clone_to` клонирует обёртку и делит compiled fn (живёт в кэше компилятора всё время процесса) — `defvar prog (HLO-COMPILE ...)` безопасен после сброса scratch.

## 19.9 HLO-пайплайн

Граф из узлов `HloOp` (см. `src/hlo/graph.hpp`): `PARAMETER, CONSTANT, DOT, ADD, MUL, RELU, FUSION, TUPLE, OUTPUT, LOOKUP, ARGMAX, SOFTMAX, COMPOSITE, CODEGEN, WEIGHTED_LOOKUP, ROUTE` (ROUTE — v2.3.5: branch-подграфы, OR-overlap-метки, per-sample taken внутри ноды; clone/merge на pointer-identity maps, DCE держит ветви живыми, CSE не сливает ROUTE-ноды).

Оптимизации: CSE, DCE, fusion поэлементных цепочек, broadcast-shape, COMPOSITE-expansion. Кодоген (`src/codegen/hlo_codegen.cpp`): C-ABI-функция `hlo_fn(params, output)` → LLVM IR → object → `dlopen` (AOT, **не OrcJIT**).

С **v2.3.0** путь **полностью in-process** (коммит `ef4994e`): `TargetMachine::emit` → `lld::elf::link` → `dlopen`. Никакого `popen`/`system`. Детект наличия LLD через CMake (`QLISP_HAS_LLD`). Если LLD не подключена при сборке — fallback на shell-линковку `popen("llc … && g++ -shared …")`. Линкеры Debian-style требуют явный `-lz -lzstd`; CI обновлён: `lld-22 liblld-22-dev`.

Кэш: `/tmp/qlisp_hlo_cache/` с versioned fingerprint (с v2.3.6 включает branch-подграфы и метки — графы с одинаковым скелетом, но разными ветвями, не делят запись кэша). Fallback: если AOT не удался — интерпретация клонированного графа (`HloGraph::clone()`); `HLO-RUN` с v2.3.7 защищён arg-count guard (ошибка вместо SIGSEGV).

С v2.3.7 граф — данные: `GRAPH-DATA`/`GRAPH-FROM-DATA` (S-выражение с ROUTE-ветвями и константами, точный круговой рейс print→read→print) и `GRAPH-RUN-PASSES` (CSE/DCE/Fusion из Lisp); COMPOSITE-ноды ездят через данные и материализуются fallback-исполнителем.

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
- С v2.3.4 роутер работает и поверх макро-слотов (`ns-route`/`ns-route-sample`/`ns-route-index`): NS-RL-движок susuwatari (T1–T4) живёт на уровне stdlib + примитивов `SAMPLE`/`EVAL-SANDBOXED` — гл. 14.10.

## 19.11 LSP

`qlisp-lsp` (`src/lsp/server.cpp`, отдельная цель CMake без LLVM и без линковки движка) — с v2.3.8 полноценный Language Server, автономный (JSON-RPC over stdio, встроенный JSON-парсер с честным анескейпингом `\n \t \r \b \f \uXXXX` — старый `json_get` терял `\n`, склеивая документ в одну строку):

- **Ядро**: двухпроходный сканер S-форм, повторяющий конвенции ридера (строки с экранированием, `;`-комментарии, диспетч-макросы `#' #(` `#\`, quote/quasiquote/comma); инкрементальная синхронизация (`change: 2`) через офсетный сплайсинг — без перечитывания документа.
- **Диагностики**: незакрытая форма (с позицией открывающей скобки), лишняя `)`, незакрытая строка — точные диапазоны.
- **Воркспейс-индекс** на initialize: скан `*.qlsp` (глубина ≤8, кап 500 файлов/1 МБ, пропуск `.git/build*/node_modules/.cache`); didSave/didClose перечитывают файл с диска.
- **Навигация**: cross-file definition (Location | Location[], кап 20), references (кап 200), documentHighlight, rename (WorkspaceEdit.changes по индексу, кап 50 URI), workspace/symbol (кап 100), documentSymbol (14 def-видов + deftest со строковым именем); сопоставление символов без учёта регистра (ридер апикасит).
- **Комфорт**: foldingRange (пары OPEN/CLOSE на разных строках), formatting (Lisp-отступы: тело defun/let/if/… = col+2, аргументы после головы headCol+len+1; многострочные строки и пустые строки не трогаются; минимальные per-line TextEdits), ~140 автодополнений из `lsp_tables.inc` с hover-документацией.
- **Ограничения (2.3.9+)**: семантические диагностики (unbound vars, арность — нужен eval-контекст), semantic tokens, signatureHelp; колонки в байтах (не UTF-16).
- Регрессионный тест — `tests/lsp_test.sh` (JSON-RPC-драйвер over stdio, 35 проверок).

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

1. **CI** (`.github/workflows/build.yml`, job `windows`): MSYS2 MinGW64 + LLVM/OpenBLAS нативно под Win64 ABI. Публикация релиза автоматическая при пуше тега (`git tag vX.Y.Z && git push origin main vX.Y.Z`); артефакты CI переиздаются в релизах документационного репозитория.
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
| Тесты | сюит зелёный (Release, Linux); ASan чисто на HLO/NS/tape-путях; soak-харнесс без роста RSS |
| LSP | инкрементальный sync (change:2), cross-file def/refs/rename, folding, formatting (v2.3.8) |

## 19.14 Известные ограничения (см. TEMP_ISSUES.md и TODO.md)

- **CUDA** (`src/gpu/kernels.cu`) не зарегистрирован и не тестирован.
- **HLO-AOT** через `popen` (не OrcJIT) — fallback, если LLD не подключена при сборке. С v2.3.0 по умолчанию **in-process LLD** (`ef4994e`).
- **EnvFrame** использует C++ `std::map` — будет переписан на SExpr-alist в Phase 12.
- **`Tensor::is_stable`** и **`Tensor::size_bytes`** пока не реализованы.
- **Произвольное переписывание графа** (`graph-node`/`graph-rewrite`) — в плане; база готова: `GRAPH-DATA`/`GRAPH-FROM-DATA`/`GRAPH-RUN-PASSES` (v2.3.7).
- **LLVM-диспетчер для ROUTE** — отложен (roadmap 1.2); ROUTE исполняется fallback-программой (сама ROUTE-нода и её backward `HLO-ROUTE-GRAD!` реализованы в 2.3.5–2.3.6).
- **HLO fallback-граф (без `HLO-COMPILE`)** — держит константы формы трассировки, кросс-форменное использование небезопасно (`TEMP_ISSUES` #23).
- **Кросс-компиляция `qlisp`/`qlispc` под Windows из Linux** — невозможна (ABI-mix SysV/Win64). Используйте CI или MSYS2.
- **Семантические диагностики LSP** (unbound vars, арность), semantic tokens, signatureHelp — roadmap 2.3.9+.
- **CI** — GitHub Actions (`build.yml`) на Linux и Windows; тесты — `.qlsp`-скрипты в `test/` (standalone) и `tests/` (`deftest`/`run-tests`); артефакты CI публикуются как релизные бинарники.
- Исторические фейлы ранних 2.3.x (`ns-nested-breed-guilty`, `neurosym-rule-check`) устранены в 2.3.1–2.3.2 (см. гл. 7.8 и 14.4).