# Глава 6. Тензоры

Тензоры — N-мерные массивы, центральный тип данных QLISP-ML. Аналог `np.ndarray` / `torch.Tensor`, но с линейной семантикой владения (гл. 4–5).

## 6.1 Создание

```lisp
;; Литерал — вложенные списки
(defvar a (tensor ((1.0 2.0) (3.0 4.0))))     ;; 2x2

;; Фабрики
(zeros (list 2 3))        ;; нули            np.zeros((2,3))
(ones (list 4 2))         ;; единицы         np.ones((4,2))
(randn (list 100 1))      ;; N(0,1)          np.random.randn(100,1)
(randint 1 6 (list 3 3))  ;; случайные целые

;; Из списка
(list->tensor '(1.0 2.0 3.0))             ;; вектор [3]
```

> 🐍 **Python-аналогия.**
> ```python
> a = np.array([[1., 2.], [3., 4.]])
> np.zeros((2, 3)); np.ones((4, 2)); np.random.randn(100, 1)
> ```

## 6.2 Информация о тензоре

```lisp
(tensor-shape a)      ;; → (2 2)        a.shape
(tref a 0)            ;; элемент по линейному индексу   a.flat[0]
(tensor-item t)       ;; скаляр из тензора размерности 1  t.item()
```

## 6.3 Арифметика и матричные операции

| QLISP | Что делает | NumPy |
|---|---|---|
| `(T+ a b)` | поэлементное сложение | `a + b` |
| `(T- a b)` | поэлементная разность | `a - b` |
| `(T* a b)` | поэлементное умножение | `a * b` |
| `(MATMUL a b)` | матричное умножение | `a @ b` |
| `(RESHAPE t (list 4 1))` | изменение формы (глубокая копия) | `t.reshape(4,1)` |
| `(TRANSPOSE t)` | транспонирование (глубокая копия) | `t.T` |
| `(STACK ...)` / `(CONCAT ...)` | сборка тензоров | `np.stack` / `np.concatenate` |
| `(T-FMA a b c)` | fused multiply-add: `a*b+c` | — (один SIMD-кирнел) |
| `(T-FMA-RELU a b c)` | `max(0, a*b+c)` | — |

Элементарные функции работают поэлементно: `(SQRT t)`, `(EXP t)`, `(LOG t)`, `(SIN t)`, `(COS t)`, `(ABS t)`, `(POW t p)`.

Арифметика `+`, `-`, `*` тоже работает с тензорами (броадкаст, как в NumPy).

```lisp
(defvar x (randn (list 3 2)))
(defvar y (randn (list 2 4)))
(print (tensor-shape (matmul x y)))    ;; → (3 4)
```

## 6.4 Broadcasting

Как в NumPy: операции растягивают измерения размера 1.

```lisp
(T+ (tensor ((1.0 2.0) (3.0 4.0))) (tensor ((10.0))))   ;; прибавит 10 ко всем
(T+ (randn (list 4 3)) (randn (list 3)))                ;; строка-вектор ко всем строкам
```

## 6.5 Редукции

```lisp
(TSUM t)                    ;; сумма всех элементов         t.sum()
(TSUM t (list 0))           ;; сумма по оси 0               t.sum(axis=0)
(TMEAN t)                   ;; среднее                      t.mean()
(ARGMAX t)                  ;; индексы максимумов           t.argmax()
```

## 6.6 Активации и элементы нейросетей

```lisp
(RELU t)      (SOFTMAX t)   (SIGMOID t)   (TANH t)
(CONV2D input kernel stride padding)
(MAXPOOL2D input kernel stride)
(DROPOUT t rate)  (BATCHNORM t)  (LAYERNORM t)
(WEIGHTED-LOOKUP indices weight-matrix)   ;; embedding-lookup
(FFT signal)                              ;; быстрое преобразование Фурье
```

> 🐍 Это `torch.nn.functional.relu / softmax / sigmoid / tanh / conv2d / max_pool2d / dropout / batch_norm / layer_norm / embedding / fft` — но как обычные функции языка, без классов-обёрток.

## 6.7 Производительность: SIMD и BLAS

- Поэлементные операции (`T+`, `T*`, `RELU`, …) векторизованы: AVX2 (8 float за инструкцию) на x86_64, NEON (4 float) на ARM.
- `MATMUL` идёт через OpenBLAS (`cblas_sgemm`) — производительность уровня NumPy/PyTorch на CPU.
- Тензоры хранятся в device-пуле; буферы переиспользуются между итерациями (гл. 5).

## 6.8 Сериализация

```lisp
(SAVE-NPY t "weights.npy")    ;; NumPy-формат!      np.save
(defvar w (LOAD-NPY "weights.npy"))               ;; np.load
```

Обмен с Python-экосистемой «из коробки»: обучили веса в QLISP — загрузили в NumPy, и наоборот.

## 6.9 Мини-пример: нормализация матрицы

```lisp
(defvar x (randn (list 5 3)))

;; z-score по каждому элементу
(defvar mu  (tmean x))
(defvar sd  (sqrt (t* (t- x mu) (t- x mu))))
(print (tensor-shape x))
```

Полные примеры с обучением — в главах 7, 8 и 17.
