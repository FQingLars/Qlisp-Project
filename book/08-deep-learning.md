# Глава 8. Глубокое обучение

Глава про построение нейросетей: слои, композиция, CNN-операции, нормализация и полный цикл обучения.

## 8.1 Слои как замыкания (модуль DL)

В QLISP нет класса `nn.Module`. Слой — это **функция, возвращающая замыкание** с захваченными параметрами:

```lisp
(import DL)

(defvar layer (linear 4 2))    ;; Linear(4 -> 2) с обучаемыми w, b

;; применение
(defvar out (layer x))         ;; x: тензор [N, 4] → [N, 2]
```

Что происходит внутри `linear`:

```lisp
(defun linear (in-features out-features)
  (let ((w (param (randn (list out-features in-features))))
        (b (param (randn (list out-features)))))
    (lambda (x)
      (t+! (matmul! w x) b))))
```

> 🐍 **Python-аналогия.** Это `nn.Linear(4, 2)` без класса: параметры живут в замыкании, а `layer(x)` — это `forward`. Функциональный стиль в духе JAX/Haiku: те же идеи, но с мутабельными на месте параметрами.

## 8.2 Композиция: `sequential`

```lisp
(defvar model (sequential (list
  (linear 2 4)     ;; скрытый слой
  (linear 4 1))))  ;; выход

(defvar out (model x))
```

> 🐍 `nn.Sequential(nn.Linear(2,4), nn.Linear(4,1))`.

Между слоями вручную вставляются активации (гл. 6–7): `(relu! (model x))`.

## 8.3 MLP для XOR: минимальная сеть

```lisp
;; данные XOR
(defvar X (tensor ((0.0 0.0) (0.0 1.0) (1.0 0.0) (1.0 1.0))))
(defvar Y (tensor ((0.0) (1.0) (1.0) (0.0))))

;; параметры: 2 → 4 (ReLU) → 1
(defvar w1 (param (randn (list 2 4))))
(defvar b1 (param (zeros (list 4))))
(defvar w2 (param (randn (list 4 1))))
(defvar b2 (param (zeros (list 1))))

(defun forward (x)
  (let ((h (relu! (t+! (matmul! x w1) b1))))
    (matmul! h w2)))

;; обучение
(defvar lr 0.5)
(defvar epoch 0)
(while (< epoch 500)
  (let ((loss (mse! (forward X) Y)))
    (grad! loss)
    (sgd-step w1 lr) (sgd-step b1 lr)
    (sgd-step w2 lr) (sgd-step b2 lr)
    (setq epoch (+ epoch 1))))

(print (forward X))   ;; ≈ (0 1 1 0)
```

Полная версия с HLO-компиляцией инференса — в главе 17.

## 8.4 CNN-операции

```lisp
;; Свёртка: вход [N,H,W,C-стиль по факту плоский], ядро, stride, padding
(defvar out (conv2d! input kernel 1 0))
(defvar pooled (maxpool2d! input 2 2))     ;; окно 2, stride 2
```

> 🐍 `F.conv2d(input, kernel, stride=1, padding=0)` и `F.max_pool2d(input, 2, 2)`.

## 8.5 Нормализация и регуляризация

```lisp
(batchnorm! h)     ;; BatchNorm
(layernorm! h)     ;; LayerNorm — стандарт для трансформеров
(dropout! h 0.5)   ;; Dropout с вероятностью 0.5 (только train!)
```

> 🐍 `nn.BatchNorm1d`, `nn.LayerNorm`, `nn.Dropout(0.5)` — только в виде функций; отключать dropout «на инференсе» нужно самому (просто не вызывать `dropout!`).

## 8.6 Функции потерь

```lisp
(mse! pred y-true)                      ;; регрессия
(cross-entropy! logits labels)          ;; классификация
(softmax! logits)                       ;; если нужен вероятностный выход
```

## 8.7 Типичный цикл обучения (шаблон)

```lisp
;; параметры
(defvar w (param (randn (list d-in d-out))))
(defvar lr 0.01)

;; функция шага
(defun step (i n)
  (if (> i n) nil
    (progn
      (let ((loss (mse! (matmul! x w) y)))
        (grad! loss)
        (sgd-step w lr)
        (if (equal (mod i 100) 0) (print loss))
        (step (+ i 1) n)))))

(step 1 1000)
```

Совет: оборачивайте счётчик эпох в рекурсию (`step i n`) — это идиоматичный QLISP-цикл; `while` тоже работает (гл. 3).

## 8.8 Логирование обучения: модуль VISUAL

```lisp
(import VISUAL)
(dashboard 8765)            ;; веб-дашборд (TensorBoard-подобный)
(log-loss loss epoch)       ;; кривая loss
(log-accuracy acc epoch)    ;; кривая accuracy
(log-tensor w "weights")    ;; heatmap весов
```

Открывайте `http://localhost:8765` в браузере во время обучения.
