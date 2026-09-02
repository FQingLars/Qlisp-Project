# Глава 8. Глубокое обучение

Глава про построение нейросетей: слои, композиция, CNN-операции, нормализация и полный цикл обучения.

## 8.1 Слои как замыкания

В QLISP нет класса `nn.Module`. Слой — это **функция, возвращающая замыкание** с захваченными параметрами. Модуль `dl` (24 строки в `stdlib/dl.qlsp`) задаёт его так:

```lisp
;; ВНИМАНИЕ: linear в stdlib/dl.qlsp использует соглашение matmul(w, x),
;; а не matmul(x, w) — см. исходник.
(defun linear (in-features out-features)
  (let ((w (param (randn (list out-features in-features))))
        (b (param (randn (list out-features)))))
    (lambda (x)
      (t+! (matmul! w x) b))))

(defvar layer (linear 4 2))    ;; Linear(4 -> 2) с обучаемыми w, b

;; применение
(defvar out (layer x))         ;; x: тензор [N, 4] → [N, 2]
```

Если вы предпочитаете `matmul(x, w)` (как в PyTorch), перепишите `linear` локально: 4 строки, никакого волшебства за `nn.Module` нет.

```lisp
;; Альтернативная версия linear (matmul(x, w))
(defun linear (in-features out-features)
  (let ((w (param (randn (list in-features out-features))))
        (b (param (randn (list out-features)))))
    (lambda (x)
      (t+! (matmul! x w) b))))
```

> 🐍 **Python-аналогия.** Это `nn.Linear(4, 2)` без класса: параметры живут в замыкании, а `layer(x)` — это `forward`. Функциональный стиль в духе JAX/Haiku: те же идеи, но с мутабельными на месте параметрами.

## 8.2 Композиция: `sequential`

`stdlib/dl.qlsp`:

```lisp
(defun sequential layers
  (lambda (x)
    (let ((out x))
      (dolist layer layers
        (setq out (layer out)))
      out)))
```

`dolist` — стандартная форма для итерации по списку. Идиоматичный вариант без `dolist`:

```lisp
(defun apply-layers (layers x)
  (if (null layers) x
    (apply-layers (cdr layers) (funcall (car layers) x))))

(defun sequential (layers)
  (lambda (x) (apply-layers layers x)))

(defvar model (sequential (list
  (linear 2 4)     ;; скрытый слой
  (linear 4 1))))  ;; выход

(defvar out (model x))
```

## 8.3 Train step

В `stdlib/dl.qlsp` есть готовая функция `train-step`:

```lisp
(defun train-step (model x y-true optimizer lr)
  (let ((y-pred (model x))
        (loss (mse! y-pred y-true)))
    (grad! loss)
    (optimizer lr x)
    (print loss)))
```

Здесь `optimizer` — функция вида `(lambda (lr x) (sgd-step w lr) (sgd-step b lr) …)` для конкретной модели.

## 8.4 Импорт

Модуль `DL` вшит в бинарник наравне с остальными восемью (`src/main.cpp:embedded_modules[]`):

```lisp
(import DL)
```

Если в вашей сборке `dl.qlsp` отсутствует (старая сборка), скопируйте содержимое `stdlib/dl.qlsp` в начало своего файла.

## 8.5 CNN-операции и нормализация

Эти операции живут не в `DL`, а в ядре (см. гл. 6) и доступны всегда:

```lisp
(CONV2D! input kernel 1 0)              ;; [N,C,H,W] x [out_C,in_C,kH,kW], stride=1, padding=0
(MAXPOOL2D! input 2 2 2)                ;; (x kh kw stride): окно 2×2, stride 2
(BATCHNORM! x gamma beta 1e-5)          ;; BN(x, γ, β, ε)
(LAYERNORM! x gamma beta 1e-5)          ;; LN(x, γ, β, ε)
(DROPOUT! h 0.5)                        ;; dropout в режиме обучения
(SOFTMAX! t)   (SIGMOID! t)   (TANH! t) ;; активации
```

Все эти операции — autograd-варианты (с `!`), запись на ленту.

## 8.6 Полный цикл обучения (XOR)

```lisp
(import DL)
(defvar lr 0.5)

(defvar X (tensor ((0.0 0.0) (0.0 1.0) (1.0 0.0) (1.0 1.0))))
(defvar Y (tensor ((0.0) (1.0) (1.0) (0.0))))

;; параметры (в замыканиях слоёв)
(defvar model (sequential (list
  (linear 2 4)        ;; 2 -> 4
  (linear 4 1))))     ;; 4 -> 1

;; ручной шаг обучения — здесь `optimizer` должен достать параметры модели
;; (для демонстрации используем sgd-step на конкретных w):
(defun train-step ()
  (let ((out (model X)))
    (let ((loss (mse! out Y)))
      (grad! loss)
      ;; ... здесь шаги SGD по всем параметрам модели
      loss)))
```

Подробный разбор полного MLP для XOR — в главе 17.

## 8.7 Стиль кода: слои vs классы

Важно понимать идиоматику ML-кода QLISP: **модели обычно пишут не классами, а замыканиями**:

```lisp
(defvar model (linear 4 2))     ;; замыкание с параметрами внутри
(model x)
```

Классы (гл. 13) уместны для **данных** (датасеты, конфиги, узлы деревьев), замыкания — для **поведения** (слои, модели). Это противоположно PyTorch, где `nn.Module` — класс.

## 8.8 Сводная таблица

| QLISP | PyTorch |
|---|---|
| `(defun linear (in out) (lambda (x) ...))` | `nn.Linear(in, out)` |
| `(sequential (list l1 l2))` | `nn.Sequential(l1, l2)` |
| `(layer x)` | `model(x)` / `forward` |
| `(BATCHNORM! h)` / `(LAYERNORM! h)` | `nn.BatchNorm1d` / `nn.LayerNorm` |
| `(DROPOUT! h 0.5)` | `nn.Dropout(0.5)` |
| `(CONV2D! x k s p)` | `F.conv2d(x, k, stride=s, padding=p)` |
| `(MAXPOOL2D! x k s)` | `F.max_pool2d(x, k, s)` |
| `(WEIGHTED-LOOKUP probs v1 v2)` | взвешенная сумма (скаляр) |