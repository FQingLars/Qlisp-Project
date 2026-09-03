# Глава 10. HLO-компиляция: от интерпретации к машинному коду

HLO-пайплайн — killer feature QLISP: ваш тензорный код трассируется в граф, оптимизируется и компилируется в **нативный машинный код** через LLVM. Это собственная реализация идей XLA/torch.compile.

## 10.1 Идея

```text
Код QLISP → трассировка в HLO-граф (START-TRACE … HLO-COMPILE)
           → оптимизации (CSE, DCE, fusion, broadcast-shape, COMPOSITE-expansion)
           → LLVM IR → object file → lldELF → .so → dlopen → нативное исполнение
```

С **v2.3.0** компиляция HLO-кернелов **полностью in-process** (коммит `ef4994e`): LLVM-эмит и линковка LLD происходят в самом процессе qlisp. Для Linux-линковки используется **in-process `lldELF`** (детект через CMake, `QLISP_HAS_LLD`). Пайплайн:

```text
HloGraph → LLVM IR (TargetMachine::emit) → object (.o)
        → in-process lldELF → .so
        → dlopen → нативный HLOPROG
```

До v2.3.0 путь был через `popen("llc … && g++ -shared …")` — теперь это fallback, если `lld` не подключена при сборке. In-process версия быстрее (нет fork/exec на каждый кёрнел) и надёжнее (нет проблем с PATH/shell-эскейпингом на Windows).

- **CSE** — устранение общих подвыражений (`HloGraph::eliminate_common_subexpressions`).
- **DCE** — удаление мёртвого кода (`HloGraph::eliminate_dead_code`).
- **Fusion** — цепочки поэлементных операций (ADD→MUL→RELU) сливаются в **один кирнел** (`HloGraph::fuse_elementwise_ops`).
- **COMPOSITE-expansion** — `defhloop`-узлы разворачиваются в mini-граф (`composite_expanded` флаг защищает от двойного разворачивания).
- **Broadcast-shape** — `add`/`mul` вычисляют numpy-style broadcast shape (right-aligned, dim=1 broadcasts), а не форму левого операнда.
- **BLAS** — узлы `DOT` (матричное умножение) вызывают `cblas_sgemm` через external LLVM declaration.
- **Кэш** — диск в `/tmp/qlisp_hlo_cache/` с versioned fingerprint (`HloCodeGen::cache_` + `cache_dir()`). Повторные запуски не перекомпилируются.
- **In-process LLD** — компилятор линкует .so через `lld::elf::link` (без `system("ld ...")`). Требует линковки `lld` + `zlib` + `zstd` (Debian-style линкеры требуют явный `-lz -lzstd`; CI обновлён: `lld-22 liblld-22-dev`).

> 🐍 **Python-аналогия.** `torch.compile(model)` / `jax.jit(f)` — те же три этапа: трассировка, оптимизация, кодоген. Разница: в QLISP это встроено в язык и работает на S-выражениях, а не на байткоде Python. С v2.3.0 — никакого `subprocess.run(["llc", ...])` в горячем пути.

## 10.2 Граф и его узлы

Из `src/hlo/graph.hpp`:

```cpp
enum class HloOp : uint8_t {
    PARAMETER, CONSTANT, DOT, ADD, MUL, RELU, FUSION, TUPLE,
    OUTPUT, LOOKUP, ARGMAX, SOFTMAX, COMPOSITE, CODEGEN,
    WEIGHTED_LOOKUP
};

struct HloNode {
    int id;
    HloOp op;
    std::vector<size_t> shape;
    std::vector<HloNode*> inputs;
    Tensor *constant_value;
    std::vector<HloOp> fused_ops;     // FUSION: последовательность операций
    std::vector<float> lookup_values; // LOOKUP / WEIGHTED_LOOKUP
    std::string composite_name;       // COMPOSITE: имя defhloop
    std::string codegen_kind;         // CODEGEN: kind ядра
    bool composite_expanded;
};
```

`HloGraph` владеет всеми узлами, поддерживает `entry()`, `dump()`, `clone()` (глубокое копирование для fallback-интерпретации), `fingerprint()` (для кэша), `optimize()` (CSE+DCE+fusion+DCE).

## 10.3 Ручная трассировка: базовый workflow

Примитивы (`src/eval/interpreter.cpp::register_primitives`):

```lisp
;; 1. Включить трассировку
(START-TRACE)

;; 2. Пометить входы как параметры графа (порядок = порядок аргументов HLO-RUN)
(defvar dx (GRAPH-PARAM x))
(defvar dw (GRAPH-PARAM w))

;; 3. Построить вычисление обычными операциями (без !)
(defvar out (T+ (MATMUL dx dw) b))

;; 4. Посмотреть граф (опционально)
(DUMP-GRAPH)

;; 5. Скомпилировать (возвращает HLOPROG — тег SExpr)
(defvar hlo (HLO-COMPILE out))

;; 6. Выполнить нативно: параметры передаются в порядке определения GRAPH-PARAM
(defvar result (HLO-RUN hlo x w b))
```

> 🐍 Аналог:
> ```python
> @torch.compile
> def f(x, w, b): return x @ w + b
> f(x, w, b)
> ```
> В QLISP трассировка явная: `GRAPH-PARAM` ≈ декларация сигнатуры графа, `HLO-RUN` ≈ вызов скомпилированной функции. Конвенция: **порядок вызовов `GRAPH-PARAM` = порядок аргументов `HLO-RUN`** (см. `TEMP_ISSUES.md` #13).

## 10.4 Два режима: трассировка и интерпретация

- Внутри `(START-TRACE)` … `(STOP-TRACE)` операции строят граф вместо вычисления.
- Вне трассировки те же операции считаются интерпретатором (с SIMD/BLAS).
- `(HLO-RUN hlo ...)` выполняет скомпилированный граф; если AOT-компиляция (`llc → g++ -shared → dlopen`) не удалась — автоматический fallback на интерпретацию клонированного графа (`HloGraph::clone()` глубокое копирование — см. `TEMP_ISSUES.md` #23).

Готовые HLO-примитивы (для построения графов без тензорных значений):

```lisp
(HLO-SOFTMAX ...)             ;; softmax-узел
(HLO-ARGMAX ...)              ;; argmax-узел
(HLO-LOOKUP ...)              ;; embedding lookup
(HLO-WEIGHTED-LOOKUP ...)     ;; взвешенная сумма по embedding'ам
(HLO-TRAIN-SGD ...)           ;; обучающий шаг прямо в скомпилированном графе
```

## 10.5 `defun` vs `defuse` vs `defuse!`

Три способа объявить функцию различаются тем, должен ли компилятор слить её тело в fused-кирнел:

| Форма | Fusion? | Возврат обязателен? | Если не слилось |
|---|---|---|---|
| `defun` | никогда | нет (можно процедуру) | — |
| `defuse` | пытается | да | молча становится обычной функцией |
| `defuse!` | принудительно | да | **ошибка компиляции** |

```lisp
(defun plain-proc (x) (print x))          ;; процедура: ради побочного эффекта

(defuse soft-fn (x) (T* x x))             ;; попробует фьюзить

(defuse! hard-fn (x) (T+ x (T* x x)))     ;; обязан: x + x*x одним кирнелом
```

> 🐍 `defuse!` ≈ `@torch.compile(fullgraph=True)` — «либо скомпилируй целиком, либо падай»; `defuse` ≈ обычный `@torch.compile` с тихим fallback.

При трассировке вызов `defuse`-функции вставляет в граф **COMPOSITE-узел** — тело функции целиком становится одним кирнелом (через `prim_hlo_compile → COMPOSITE-expansion`).

## 10.6 Range-путь HLO (многопоточный)

`HloCodeGen::compile_to_fn` строит два варианта функции:

```cpp
if (can_range) compile_to_fn(name + "_range", graph, module.get(), need_cblas, true);
```

Range-функция: `void fn(float** params, float* output, i64 start, i64 end)` —
- скалярный пролог (start → aligned),
- векторный цикл (aligned → end), 8 элементов/итерация,
- `!llvm.loop.parallel_accesses` metadata для авто-распараллеливания.

Range_fn генерируется **только когда entry-нода — FUSION или RELU/ADD/MUL с PARAMETER-входами** (проверка в `compile()`, см. `AGENTS.md`). Для не-fused графов range_fn пропускается (non-range путь).

## 10.7 Компиляция обученной модели: полный пример

Из `examples/linear_regression.qlsp`:

```lisp
;; ... обучили w и b автоградиентом (гл. 7) ...

;; Компилируем forward для инференса
(START-TRACE)
(defvar dx (GRAPH-PARAM x-test))
(defvar dw (GRAPH-PARAM w))
(defvar db (GRAPH-PARAM b))
(defvar dp (T+ (MATMUL dx dw) db))

(defvar hlo (HLO-COMPILE dp))
(defvar hlo-result (HLO-RUN hlo x-test w b))   ;; нативный инференс
```

Разбор целиком — в главе 17.

## 10.8 Кросс-компиляция HLO

HLO-граф целевой-агностичен: скомпилировать можно под любую платформу LLVM без пересборки QLISP:

```bash
QLISP_TARGET_TRIPLE=aarch64-linux-gnu QLISP_TARGET_CPU=cortex-a72 qlisp train.qlsp
QLISP_TARGET_TRIPLE=arm64-apple-macos  QLISP_LLC=llc-arm64 QLISP_CXX=arm64-clang++ qlisp infer.qlsp
```

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `QLISP_TARGET_TRIPLE` | host (пустая строка) | Целевой triple LLVM (`aarch64-linux-gnu`, `arm64-apple-macos`) |
| `QLISP_TARGET_CPU` | `native` | Модель CPU для `-mcpu=` (`cortex-a72`, `apple-m1`, `znver4`) |
| `QLISP_LLC` | `llc` | Путь к `llc` |
| `QLISP_CXX` | `g++` | C++-компилятор для линковки `.so` |

Эти переменные читаются через `platform::env_or` в `src/core/platform.hpp`.

## 10.9 Когда использовать HLO

- **Инференс обученной модели** — главный сценарий: граф фиксирован, fusion даёт максимум.
- **Горячие циклы обучения** — через `HLO-TRAIN-SGD` и `defuse!`-функции.
- Прототипирование — оставайтесь в интерпретаторе; компилируйте, когда алгоритм устоялся.

Практика: сначала `(defuse soft ...)`, и повышайте до `defuse!`, когда убедились, что тело трассируемое (только тензорные операции, без `print`, `if` по значениям тензоров и т.п.).

## 10.10 Известные ограничения (TODO)

- `graph-node`, `graph-rewrite` — публичные примитивы для переписывания графа из языка **не реализованы** (см. `TODO.md` §1.2 — в плане).
- ROUTE-узел (`HloOp::ROUTE`) для нейросимвольного маршрутизатора в HLO-графе **не реализован** (см. `TODO.md` §1.1 — `do_ns_if` сейчас разрывает ленту на точке выбора, а не строит ROUTE-узел).
- `composite_expanded` флаг защищает от двойного разворачивания; identity-COMPOSITE (тело = сам параметр) коллапсируется в PARAM.
- Кэш использует `std::hash` для fingerprint — теоретические коллизии, <10⁻⁹ вероятность (`TEMP_ISSUES.md` #22).