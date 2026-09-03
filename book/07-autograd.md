# Глава 7. Автоградиент

В QLISP встроен reverse-mode автоградиент через **ленту градиентов** (gradient tape) — как `torch.autograd`, но на уровне ядра языка.

## 7.1 Конвенция: операции с `!`

- **Обычные операции** (`MATMUL`, `T+`, `RELU`, …) — чистые вычисления без записи градиентов.
- **Autograd-операции** (суффикс `!`: `MATMUL!`, `T+!`, `RELU!`, …) — то же вычисление **плюс запись узла на ленту**.

> 🐍 **Python-аналогия.** В PyTorch запись на ленту управляется флагом `requires_grad` тензора. В QLISP — выбором имени операции: `MATMUL` vs `MATMUL!`. Явно, видно в коде, ничего не «сливается» случайно.

Имена autograd-примитивов (из `src/eval/interpreter.cpp:register_primitives`): `T+!`, `T*!`, `MATMUL!`, `RELU!`, `SIGMOID!`, `TANH!`, `SOFTMAX!`, `LOG!`, `EXP!`, `TSIN!`, `TCOS!`, `MSE!`, `CROSS-ENTROPY!`, `CONV2D!`, `MAXPOOL2D!`, `BATCHNORM!`, `LAYERNORM!`, `DROPOUT!`.

## 7.2 Обучаемые параметры: `PARAM`

```lisp
(defvar w (param (randn (list 4 2))))   ;; обучаемый вес
(defvar b (param (zeros (list 4))))     ;; обучаемое смещение
```

`param` (примитив `prim_param`) — оборачивает тензор как `requires_grad=true`. Аналог `torch.nn.Parameter(..., requires_grad=True)`.

## 7.3 Forward + Backward

```lisp
;; данные
(defvar x (randn (list 100 4)))
(defvar y (randn (list 100 2)))

;; модель: y = x·w + b
(defun predict (x) (t+! (matmul! x w) b))

;; forward
(let ((pred (predict x)))
  (let ((loss (mse! pred y)))     ;; MSE-loss с записью на ленту
    (grad! loss)                  ;; backward: всю цепочку — в градиенты
    (print (tensor-shape (grad-of w))))))
```

- `(grad! loss)` (примитив `prim_grad_backward`) ≈ `loss.backward()` — проход по ленте в обратном порядке (reverse-topo через `result → producer` карту в `GradientTape`), вычисление градиентов всех `PARAM`-тензоров, очистка ленты.
- `(grad-of w)` (примитив `prim_grad_of`) ≈ `w.grad`.

Внутри `GradientTape::backward(loss, arena)` идёт по dataflow-графу в reverse-topo порядке (потребители раньше производителей), накапливая `+=` градиенты во `inputs`. `GradNode::inputs` хранит `Tensor*` (не SExpr) — autograd не завязан на интерпретатор (см. `src/tensor/CODE.md`).

## 7.4 Оптимизаторы

```lisp
(sgd-step w 0.01)          ;; w -= lr * grad        (SGD) — примитив SGD-STEP
(adam-step w 0.001)        ;; Adam с состоянием m/v на тензоре — примитив ADAM-STEP
```

> 🐍 `(sgd-step w lr)` ≈ `optimizer.step()` для одного параметра. Состояние Adam (`m`, `v`, счётчик шага) хранится прямо на тензоре-параметре (`Tensor::m`, `Tensor::v`, `Tensor::step`).

Шаг обучения целиком:

```lisp
(defun train-step ()
  (let ((pred (predict x)))
    (let ((loss (mse! pred y)))
      (grad! loss)
      (sgd-step w 0.01)
      (sgd-step b 0.01)
      loss)))          ;; auto-clone: loss переживёт выход из let (гл. 5)
```

## 7.5 Поддерживаемые autograd-операции

| Категория | Операции |
|---|---|
| Арифметика | `T+!`, `T*!`, `MATMUL!` |
| Активации | `RELU!`, `SOFTMAX!`, `SIGMOID!`, `TANH!`, `EXP!`, `LOG!` |
| Loss | `MSE!`, `CROSS-ENTROPY!` |
| CNN/нормализация | `CONV2D!`, `MAXPOOL2D!`, `DROPOUT!`, `BATCHNORM!`, `LAYERNORM!` |
| Тригонометрия | `TSIN!`, `TCOS!` |
| FMA (без ленты, чистые SIMD-кирнелы) | `T-FMA`, `T-FMA-RELU` |

Градиент конкретного узла: `(GRAD-OF tensor)` (примитив `prim_grad_of`).

Ядро поддерживает и поэлементные autograd-узлы для экспоненты, логарифма и тригонометрии: `EXP!`, `LOG!`, `TSIN!`, `TCOS!`.

Градиенты в половинной точности (`F16`) работают на уровне тензорного слоя: операции и `GRAD!` не требуют промоушена, gradient seed инициализируется в dtype тензора. **HLO-путь остаётся F32-only** (см. `src/tensor/CODE.md`).

## 7.6 Сравнение с PyTorch

| PyTorch | QLISP |
|---|---|
| `w = torch.randn(4, 2, requires_grad=True)` | `(defvar w (param (randn (list 4 2))))` |
| `pred = x @ w` | `(defvar pred (matmul! x w))` |
| `loss = F.mse_loss(pred, y)` | `(defvar loss (mse! pred y))` |
| `loss.backward()` | `(grad! loss)` |
| `w.grad` | `(grad-of w)` |
| `opt.step()` | `(sgd-step w lr)` |
| `opt = Adam(..., lr=0.001); opt.step()` | `(adam-step w 0.001)` |
| `with torch.no_grad():` | обычные операции без `!` (`matmul`, `t+`, …) |

## 7.7 Полный пример: линейная регрессия

```lisp
;; y = 2x + 1
(defvar x-data (randn (list 100 1)))
(defvar y-data (t+ (t* x-data (tensor ((2.0)))) (tensor ((1.0)))))

(defvar w (param (randn (list 1 1))))
(defvar b (param (zeros (list 1 1))))

(defun train (iter max-iter)
  (if (> iter max-iter) nil
    (progn
      (let ((loss (mse! (t+! (matmul! x-data w) b) y-data)))
        (grad! loss)
        (sgd-step w 0.02)
        (sgd-step b 0.02)
        (if (equal (mod iter 50) 0) (print loss))
        (train (+ iter 1) max-iter)))))

(train 1 300)
(print w)   ;; ≈ 2.0
(print b)   ;; ≈ 1.0
```

Полный разбор этого и более сложных примеров (MLP, XOR) — в главе 17.

## 7.8 Контракт gradient tape (v2.3.0+, исправления TBC #26)

В v2.2.x канонический паттерн «граф в одной форме, `GRAD!` в следующей» опирался на то, что память нод tape ещё не переиспользована. Это **UB**: проявлялось флэйками, зависаниями, `heap-use-after-free` под ASan. В v2.3.0 контракт уточнён и жёстко зафиксирован.

### Что изменилось

1. **Tape-узлы принадлежат самой tape.** Нода аллоцируется в heap и удаляется в `tape.clear()`. До v2.3.0 они жили в scratch-блоках, которые переиспользовались между формами — это и был источник `heap-use-after-free`.

2. **`defvar` промоутит промежуточный тензор.** Когда промежуточный результат `MATMUL!` или `T+!` сохраняется через `defvar`/`setq`, его тензор клонируется в stable-scope, а ребро `result` ноды tape перенаправляется на эту stable-копию. Градиент течёт в тензор, который видит пользователь.

3. **`reset_scratch` «запечатывает» незакрытые графы.** Если форма верхнего уровня завершилась без `GRAD!`, и какие-то tape-узлы ссылаются на scratch-тензоры, эти тензоры один раз клонируются в stable, и рёбра переписываются. Обучающие циклы ничего не платят — tape пуст после каждого `GRAD!`.

4. **Adam-состояние живёт на параметре.** `m`/`v`/`step` теперь аллоцируются в stable-scope при первом `adam-step` и не умирают с итерацией. Раньше `m`/`v` создавались заново каждый шаг — это убивало momentum.

5. **HLO/NS-прокси стали строковыми хендлами.** Биндинги копируют скаляры, и fixnum-прокси терял идентичность (ключ `trace_nodes_`). Теперь — строковые хендлы, детерминированно.

### Что это значит на практике

| Сценарий | v2.2.x | v2.3.0 |
|---|---|---|
| `defvar h (matmul! x w)` затем в **следующей** форме `(grad! loss)` где `loss` зависит от `h` | флэйки, UAF под ASan | детерминированно зелёный |
| Adam на 10000 итераций | `m`/`v` умирают, momentum не работает | momentum работает корректно |
| `defvar pred (matmul! x w))`, `loss = mse! pred y`, `grad!` — в одной форме | OK | OK |
| `(setq loss (mse! pred y))` (`setq` вместо `defvar`) | OK (литералы не копируются) | OK, но `setq` всё равно копирует скаляры (v2.3.0+: биндинги не алиасят литералы) |

### `defvar_autograd` (новая конвенция)

Для backward-совместимости и читаемости — отдельная форма для тензоров-параметров, которые обязаны быть `requires_grad=true` и переживать HibLin-promote:

```lisp
(defvar_autograd w (randn (list 4 2)))
;; эквивалентно (defvar w (param (randn (list 4 2)))),
;; но явно обозначает намерение — параметр автограда.
```

(Реализовано в `stdlib/dl.qlsp` как макрос-обёртка над `defvar` + `param`.)

### Правила для авторов модулей

- Градиенты считаются **только** через `!`-операции: `MATMUL!`, `T+!`, `T*!`, `MSE!`, `RELU!`, `SIGMOID!`, `TANH!`, `SOFTMAX!`, `LOG!`, `EXP!`, `TSIN!`, `TCOS!`, `CONV2D!`, `MAXPOOL2D!`, `BATCHNORM!`, `LAYERNORM!`, `DROPOUT!`, `CROSS-ENTROPY!`. Обратной разницы «обычных» операций нет (`T-!` не существует).
- `GRAD!` = `backward` + `tape.clear()`. После `GRAD!` tape пуст.
- `GRAD-OF` читается в той же форме, где был `GRAD!` (тензоры градиентов живут в scratch текущей формы).
- Промежуточные тензоры **не стоит** держать через `defvar` (промоут клонирует тензор — лишняя копия, граф длиннее, чем нужно). Алиасы параметров через `let` остаются в scratch и не платят за копирование.
- При `defvar` параметра автограда — он переживает `reset_scratch` (тензор уже в stable), и tape-узел, на него ссылающийся, тоже.

### Что НЕ исправлено (известные ограничения)

- HLO fallback-граф (без `HLO-COMPILE`) держит константы формы трассировки — кросс-форменное использование такого HLOPROG небезопасно (`TEMP_ISSUES` #23).
- Shell-линковка HLO-кернела — fallback при сборке без LLD.

### Детерминизм тестов

Сюит v2.3.0: **48/50** (Release, Linux). ASan: чисто на tape-тяжёлых тестах (`dl`, `defvar_autograd`, `hlo_training`) — `heap-use-after-free` из 2.2.x не воспроизводится. Два детерминированных фейла, существовавших до 2.3.0 и вне скоупа релиза:

- `tests/ns_router.qlsp` → `ns-nested-breed-guilty` (guilt-блокировка в вложенном NS-IF);
- `tests/symbolic_neuro.qlsp` → `neurosym-rule-check` (символический движок правил).