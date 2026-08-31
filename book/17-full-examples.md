# Глава 17. Разбор примеров

Два канонических примера из `examples/` исходного репозитория, разобранных построчно, с Python-эквивалентами.

## 17.1 Линейная регрессия: обучение + HLO-инференс

**Задача.** Синтетические данные `y = 2x + 1 + noise`. Обучаем `w`, `b` методом MSE + SGD, затем компилируем forward в нативный код.

### Полный код

```lisp
;; ----- данные -----
(defvar x-data (RANDN (list 100 1)))
(defvar noise (T* (RANDN (list 100 1)) (tensor ((0.2)))))
(defvar y-data (T+ (T+ (T* x-data (tensor ((2.0)))) (tensor ((1.0)))) noise))

;; ----- параметры -----
(defvar w (PARAM (RANDN (list 1 1))))
(defvar b (PARAM (ZEROS (list 1 1))))
(defvar lr 0.02)

;; ----- forward -----
(defun predict (x) (T+! (MATMUL! x w) b))

;; ----- обучение -----
(defun train-loop (iter max-iter)
  (if (> iter max-iter) nil
    (progn
      (let ((yp (predict x-data))
            (loss (MSE! yp y-data)))
        (GRAD! loss)
        (SGD-STEP w lr)
        (SGD-STEP b lr)
        (if (equal (mod iter 40) 0)
          (progn (PRINT "epoch") (PRINT iter) (PRINT loss)))
        (train-loop (+ iter 1) max-iter)))))

(train-loop 1 200)

;; ----- компиляция инференса в HLO -----
(START-TRACE)
(defvar dx (GRAPH-PARAM x-test))
(defvar dw (GRAPH-PARAM w))
(defvar db (GRAPH-PARAM b))
(defvar dp (T+ (MATMUL dx dw) db))

(defvar hlo (HLO-COMPILE dp))
(defvar hlo-result (HLO-RUN hlo x-test w b))
```

### Построчный разбор

| Строки | Что происходит | Python |
|---|---|---|
| `x-data`, `noise`, `y-data` | Датасет 100×1: `y = 2x + 1 + 0.2·ε` | `x = np.random.randn(100,1); y = 2*x + 1 + 0.2*np.random.randn(100,1)` |
| `w`, `b` | Обучаемые параметры `[1,1]` | `w = nn.Parameter(torch.randn(1,1))` |
| `predict` | Forward: `x·w + b`. `!`-операции пишут граф | `def predict(x): return x @ w + b` |
| `train-loop` | Рекурсия-цикл: forward, loss, backward, шаг SGD | `for i in range(200): loss.backward(); opt.step()` |
| `(GRAD! loss)` | Backward по ленте | `loss.backward()` |
| `(SGD-STEP w lr)` | Обновление весов на месте | `w -= lr * w.grad` |
| `(START-TRACE)…(HLO-COMPILE)` | Трассировка forward в граф и компиляция | `compiled = torch.compile(lambda x: x @ w + b)` |
| `(HLO-RUN hlo ...)` | Нативный инференс | `compiled(x_test)` |

Обратите внимание: тензоры-константы записываются как `(tensor ((2.0)))` — скаляр в форме `[1,1]`, чтобы избежать сюрпризов broadcasting'а.

## 17.2 XOR: двухслойный MLP

**Задача.** Обучить сеть `2 → 4 (ReLU) → 1` решать XOR.

```lisp
;; ----- датасет -----
(defvar X (tensor ((0.0 0.0) (0.0 1.0) (1.0 0.0) (1.0 1.0))))
(defvar Y (tensor ((0.0) (1.0) (1.0) (0.0))))

;; ----- параметры -----
(defvar w1 (PARAM (RANDN (list 2 4))))
(defvar b1 (PARAM (ZEROS (list 4))))
(defvar w2 (PARAM (RANDN (list 4 1))))
(defvar b2 (PARAM (ZEROS (list 1))))
(defvar lr 0.5)

;; ----- forward -----
(defun forward (x)
  (let ((h (RELU! (T+! (MATMUL! x w1) b1))))
    (MATMUL! h w2)))

;; ----- обучение -----
(defun train-epoch (iter max-iter)
  (if (> iter max-iter) nil
    (progn
      (let ((loss (MSE! (forward X) Y)))
        (GRAD! loss)
        (SGD-STEP w1 lr) (SGD-STEP b1 lr)
        (SGD-STEP w2 lr) (SGD-STEP b2 lr)
        (if (equal (mod iter 100) 0) (progn (PRINT "epoch") (PRINT iter) (PRINT loss)))
        (train-epoch (+ iter 1) max-iter)))))

(train-epoch 1 500)

;; ----- HLO-компиляция инференса -----
(START-TRACE)
(defvar dx (GRAPH-PARAM X))
(defvar w1p (GRAPH-PARAM w1))
(defvar b1p (GRAPH-PARAM b1))
(defvar w2p (GRAPH-PARAM w2))
(defvar h (RELU (T+ (MATMUL dx w1p) b1p)))
(defvar out (MATMUL h w2p))
(defvar hlo (HLO-COMPILE out))
(defvar hlo-pred (HLO-RUN hlo X w1 b1 w2))
```

### Python-эквивалент (PyTorch)

```python
import torch, torch.nn.functional as F

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y = torch.tensor([[0.],[1.],[1.],[0.]])

w1 = torch.randn(2, 4, requires_grad=True)
b1 = torch.zeros(4, requires_grad=True)
w2 = torch.randn(4, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)
lr = 0.5

def forward(x):
    return (F.relu(x @ w1 + b1)) @ w2

for epoch in range(1, 501):
    loss = F.mse_loss(forward(X), Y)
    loss.backward()
    with torch.no_grad():
        w1 -= lr * w1.grad; b1 -= lr * b1.grad
        w2 -= lr * w2.grad; b2 -= lr * b2.grad
        w1.grad = b1.grad = w2.grad = b2.grad = None
    if epoch % 100 == 0:
        print(epoch, loss.item())
```

### Ключевые различия стилей

1. **Нет `optimizer.zero_grad()`** — лента QLISP очищается в `GRAD!` автоматически.
2. **Нет `torch.no_grad()`** — разграничение train/inference-операций делается выбором `!`/без-`!`.
3. **Рекурсия вместо `for`** — идиоматично для Лиспа, `while` тоже доступен.
4. **HLO-инференс** — отдельный этап: трассировка → компиляция → нативный вызов. В PyTorch это `torch.compile`, в QLISP — язык первого класса.

## 17.3 Идеи для упражнений

1. Добавьте Adam вместо SGD: `(ADAM-STEP w lr)` и сравните сходимость.
2. Замените MSE на `CROSS-ENTROPY!` и добавьте `SOFTMAX!` — превратите регрессор в классификатор.
3. Оберните `predict` в `defuse!` и проверьте, что HLO соберёт всё в один fused-кирнел.
4. Прогоните датасет через `(CROSS-VAL-SCORE ...)` из модуля ML.
5. Сохраните обученные веса `(SAVE-NPY w "w.npy")` и постройте их в NumPy.
