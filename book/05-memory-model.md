# Глава 5. Модель памяти: HibLin (гибридная линейная аллокация)

Модель памяти QLISP называется **HibLin** (Hybrid Linear) — и это главная идея, отличающая QLISP
и от Python (GC), и от Rust (borrow checker), и от C (malloc/free). Данные по умолчанию
**одноразовые** и живут ровно в том scope, где созданы; при выходе из scope они возвращаются
в пулы. Никакого GC, ни borrow-checker, ни ручного free.

## 5.1 Аналогия с черновиком

Решая пример в столбик на бумаге, вы пишете исходные числа на черновике, промежуточный
результат — тоже на черновике, и переписываете в чистовик только ответ. Черновик
выбрасывается.

HibLin работает так же: данные по умолчанию **одноразовые** (disposable). Сохранить можно
только явным **переписыванием** — `defvar` (в «тетрадь»), а все промежуточные значения
живут в текущем scope и исчезают вместе с ним.

> 🐍 **Python-аналогия.** В Python всё живёт в heap, и garbage collector периодически
> находит и освобождает недостижимые объекты. Вы думаете о владении редко — платя GC-паузами
> и накладными расходами. В QLISP освобождение детерминированное и мгновенное: scope
> кончился — тензорные буферы вернулись в пул, cons-ячейки тоже.

## 5.2 Три правила

| Правило | Что означает | Аналогия |
|---|---|---|
| 1. Всё disposable | Данные живут в текущем scope, при выходе возвращаются в пул | Черновик |
| 2. Explicit save | `defvar`/`setq` глобального клонирует в стабильную память | Переписал в тетрадь |
| 3. No aliasing | Один владелец у каждого значения, нет shared-ссылок | Листок нельзя скопировать, можно только переписать |

## 5.3 Три пула HibLin

В HibLin **три специализированных пула**, каждый для своего типа данных. Подробности реализации — в `src/core/`.

| Пул | Файл | Назначение | Размер объектов | Переиспользование |
|---|---|---|---|---|
| **TensorBufferPool** | `tensor_buffer_pool.cpp` | Тензоры, активации, градиенты | Точный размер (0% перерасхода) | `unordered_map<size_t, vector<Entry>>` по точному размеру, O(1) для повторных размеров; лимит 1 ГБ |
| **ConsCellPool** | `cons_cell_pool.cpp` | S-выражения, cons-ячейки (списки, формы) | Фиксированный (sizeof(SExpr)) | Свободный список (free list), блоки по `BLOCK_CELLS` ячеек через `platform::page_alloc` |
| **StableMemory** | `stable_memory.cpp` + `Scope(stable=true)` | Веса моделей, глобальные структуры (`defvar`/`defun`) | Любой | Не переиспользуется; живёт до завершения программы |

> 🐍 **Python-аналогия.** Это как если бы PyTorch держал три цеха: один — только
> для тензорных буферов (точно по размеру, без мусора), второй — только для маленьких
> списков, третий — для весов модели, которые не трогает до конца. Ни один байт не
> пропадает, и ничто не «собирается» через раз.

### Детали TensorBufferPool

```cpp
// src/core/tensor_buffer_pool.hpp
class TensorBufferPool {
    std::unordered_map<size_t, std::vector<Entry>> pools_;  // exact-size buckets
    size_t pooled_ = 0;
    size_t max_pool_size_ = 1024 * 1024 * 1024;  // default 1 GB
    static void *pool_alloc(size_t);   // platform::page_alloc (mmap / VirtualAlloc)
    static void pool_free(void *, size_t);
};
```

- **Exact-size**: ключ в `unordered_map` — точный размер в байтах. Никаких size-class'ов, никакого округления.
- **Acquire** (`TensorBufferPool::acquire(bytes)`): если в bucket'е нужного размера есть свободный буфер — pop O(1); иначе `pool_alloc(bytes)` через mmap/VirtualAlloc.
- **Release** (`TensorBufferPool::release(ptr, size)`): если `pooled_ + size <= max_pool_size_` — push в bucket; иначе — `pool_free` (mmap-регион возвращается ОС).
- **Thread-safe**: `std::mutex` вокруг `unordered_map`.

Это полностью соответствует идее HibLin, описанной в `ARCHITECTURE.md` главы 5 (точное выделение, без арен и size-class'ов). Замечание: в `src/tensor/CODE.md` ошибочно упомянуты «power-of-2 buckets» — фактическая реализация exact-size (`src/core/tensor_buffer_pool.cpp:12-23`).

### Детали ConsCellPool

```cpp
// src/core/cons_cell_pool.cpp
SExpr *ConsCellPool::acquire() {
    if (free_list_.empty()) grow();           // аллокация блока через page_alloc
    SExpr *cell = free_list_.back();
    free_list_.pop_back();
    *cell = SExpr();                            // zero-initialize (tag = NIL)
    return cell;
}
```

Блоки растут по мере необходимости; при `clear()` все блоки возвращаются ОС через `platform::page_free`.

### Детали StableMemory и Scope

`Scope` (`src/core/scope.hpp`) — единица времени жизни с двумя режимами:

| Режим | Не-SExpr-аллокации | Cons-ячейки |
|---|---|---|
| `stable=true` (root/immortal) | идут в StableMemory (обычный `new`), никогда не освобождаются до teardown scope | из ConsCellPool, никогда не возвращаются |
| `stable=false` (scratch) | bump-блоки (`DEFAULT_BLOCK_SIZE = 16 МБ`), `Scope::reset()` сбрасывает | из ConsCellPool, возвращаются при reset/restore |

`Scope::track_external(ptr, bytes)` — учёт внешних (тензорных) буферов: при reset/restore они возвращаются в TensorBufferPool через `release(ptr, bytes)`.

`Scope::promote(val)` — единственный мост между мирами: вызывает `clone_to(*parent_)` (глубокое копирование в parent scope).

### Два мира (владеет Interpreter)

`Interpreter` держит два scope:

- `root_arena_` (stable=true) — для `defvar`/`defun`/`defmacro`/`defrule`/`define-compiler-macro`/plist/весов модели.
- `scratch_arena_` (stable=false) — дочерний от root; все disposable значения.

Доступы: `Interpreter::scope()` (scratch), `Interpreter::stable_scope()` (root). Примитив `SCOPE-STATS` (`prim_scope_stats`) возвращает `allocated_bytes()` текущего scope.

REPL после каждого выражения вызывает `Interpreter::reset_scratch()` — disposable-область сбрасывается, буферы тензоров и cons-ячейки возвращаются в пулы.

### `reset_scratch` в v2.3.0: «запечатывание» графа (TBC #26)

В v2.2.x паттерн «граф в одной форме, `GRAD!` в следующей» опирался на то, что память tape-узлов ещё не переиспользована. Это **UB**: проявлялось флэйками, `heap-use-after-free` под ASan. С v2.3.0 (коммит `371dd46`) `reset_scratch` стал «запечатывать» незакрытые графы:

1. Если форма верхнего уровня завершилась без `GRAD!`, и какие-то tape-узлы ссылаются на scratch-тензоры, эти тензоры **один раз клонируются в stable** (per-tensor — не на каждый узел).
2. Рёбра `result` tape-узлов переписываются на stable-копии.
3. Обучающие циклы ничего не платят — tape пуст после каждого `GRAD!`, и `reset_scratch` идёт по пустому tape.

Это означает, что `defvar` промежуточного тензора и `grad!` в **следующей** форме теперь детерминированно зелёные в Release и ASan. Подробности — в гл. 7 (раздел «Контракт gradient tape»).

### Времена жизни градиентов (v2.3.1+, TBC #28)

В v2.3.1 уточнён контракт смерти `Tensor::grad`. Раньше `backward()` аллоцировал `->grad` в scratch-регионе вызова: когда регион умирал (restore итерации `while`, сброс формы), поле указывало в переиспользованный пул — следующий backward делал `add_inplace` в чужую память (ASan heap-buffer-overflow, в release — мусорные градиенты: цикл «учился» на мусоре).

Теперь при входе в `backward()` на каждый нодовый тензор регистрируется guard (`Scope::track_dtor` + `Tensor::clear_grad`): при смерти региона поле честно обнуляется, следующая итерация аллоцирует свежий буфер. Guard'ы срабатывают LIFO по порядку аллокаций — старый guard не может затереть живой новый градиент; `seal()` не копирует на stable-клон grad умирающего региона.

Контракт `GRAD-OF` не изменился (нули при отсутствии grad — на этом держатся NS guilt-тесты), но истёкший grad теперь настоящий null, а не указатель в переиспользованный пул.

## 5.4 Два мира данных

```text
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  SCRATCH (disposable)        │     │  STABLE (persistent)        │
│                              │     │                              │
│  let-биндинги                │     │  defvar/defun/defmacro       │
│  промежуточные SExpr         │     │  глобальные биндинги         │
│  временные списки            │     │  код-как-данные:             │
│  локальные замыкания         │     │    макросы, правила          │
│  forward/backward тензоры    │     │  plist-данные                │
│                              │     │  веса модели                 │
│                              │     │                              │
│  Живёт 1 scope.              │     │  Живёт всю программу.        │
│  Выход — всё в пулы.         │     │  Никогда не сбрасывается.    │
└─────────────────────────────┘     └─────────────────────────────┘
```

Мост между мирами: `defvar`/`setq` глобального — значение **глубоко клонируется**
в стабильный мир (`Scope::promote`). Тензор — новый буфер в StableMemory; список — новые ячейки; ни одна ссылка не переходит из scratch в stable.

## 5.5 Операции и их владение

| Операция | Входы | Выход | Где живёт |
|---|---|---|---|
| `(MATMUL x w)` | `x`, `w` читаются (не потребляются) | новый тензор | scratch (TensorBufferPool) |
| `(defvar w val)` | `val` нетронут | копия `val` (clone_to) | stable (root scope) |
| `(setq w val)` глобальной | — | — | обновляет `w` на месте *и* клонирует val |
| `(setq x val)` локальной | — | — | обновляет `x` на месте в scratch |
| `(defun f ...)` | — | — | замыкание → root scope (через `root_arena_`) |
| `(defmacro m ...)` | — | — | замыкание → root scope |
| `(defrule name (vars) (when ...) (assert ...))` | — | — | правило → root scope (clone_to) |
| `(assert-fact ...)` | — | — | факт → root scope |
| `return result` (через `progn`/let) | — | — | auto-clone в scope-родителя при `promote` |

Техническая деталь: тензор в scratch живёт в TensorBufferPool; при выходе из scope
буфер возвращается в точный free list — следующий тензор того же размера возьмёт его
за O(1), без системного аллокатора.

## 5.6 Почему это идеально для ML

Типичный цикл обучения:

```lisp
(while (< i 1000)
  (let ((pred (matmul! x w)))       ; pred ~1 МБ на scratch
    (let ((loss (mse! pred y)))     ; loss ~1 байт, pred потреблён
      (grad! loss)                  ; градиенты на scratch
      (sgd-step w 0.01)             ; w обновлён на месте (stable)
      (setq i (+ i 1)))))

1000 итераций используют **одни и те же ~1 МБ** scratch-памяти: без единого free,
без GC-пауз, без роста потребления. Каждая итерация while создаёт новый scope
через `checkpoint()/restore()` (см. `Scope::checkpoint()`, `Scope::restore()`); на
выходе итерации буферы возвращаются в пулы.

Кстати, `while` с v2.3.3 (коммит `a412beb`) **не растёт в памяти вовсе**: раньше `do_while` каждый ход продвигал результат тела в immortal root arena и не освобождал предыдущую копию (~30 МБ на 1M итераций). Теперь результат живёт в паре ping-pong scratch-скоупов (максимум две живые копии), в root продвигается один раз на выходе — те же семантики возврата, O(1) памяти; попутно циклы ускорились ~2× (loop-1m 20s → 10s). Контракт возвращаемого значения зафиксирован в `tests/while_semantics.qlsp`. Утечки памяти в ядре ловит soak-харнесс (`tests/memory_soak.qlsp` + `bench/leak_watch.py`): churn-циклы (cons, строки, eager-тензоры, autograd, HLO, while) с VmRSS-чекпоинтами через `/proc`.