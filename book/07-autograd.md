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