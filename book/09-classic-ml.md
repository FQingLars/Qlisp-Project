# Глава 9. Классический ML (модуль ML)

Модуль `ML` — встроенный «мини-scikit-learn»: утилиты, ядра, метрики, препроцессинг, KNN, наивный Байес, деревья, ансамбли, K-Means, кросс-валидация и grid search. Всё реализовано на самом QLISP (см. `stdlib/ml.qlsp`) — хороший материал для изучения языка.

```lisp
(import ML)
```

## 9.1 Утилиты для списков

```lisp
(range 5)                 ;; → (0 1 2 3 4)       range(5)
(range 5 10 2)            ;; → (5 7 9) с &optional start/step
(take 2 '(1 2 3))         ;; → (1 2)             lst[:2]
(drop 2 '(1 2 3))         ;; → (3)               lst[2:]
(unique '(1 1 2))         ;; → (1 2)             set-подобно
(mean '(1 2 3))           ;; → 2                 statistics.mean
(zip-with + '(1 2) '(10 20))  ;; поэлементный zip
```

## 9.2 Ядра (kernels)

```lisp
(linear-kernel x1 x2)            ;; ⟨x1, x2⟩
(poly-kernel x1 x2 degree coef0) ;; (⟨x1,x2⟩+c)^d
(rbf-kernel x1 x2 gamma)         ;; exp(-γ‖x1−x2‖²)
(sigmoid-kernel x1 x2 a c)       ;; tanh(⟨x1,x2⟩·a+c)
```

> 🐍 Это ядра из `sklearn.metrics.pairwise` / `sklearn.svm`.

## 9.3 Метрики

```lisp
(accuracy-score y-pred y-true)
(precision-score y-pred y-true 'positive)
(recall-score y-pred y-true 'positive)
(f1-score y-pred y-true 'positive)
(confusion-matrix y-pred y-true labels)
```

> 🐍 `sklearn.metrics.accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`.

## 9.4 Препроцессинг и разбиение

```lisp
(train-test-split data target 0.8)     ;; 80% train / 20% test
(standard-scale-lst data)              ;; z-score     StandardScaler
(minmax-scale-lst data)                ;; [0,1]       MinMaxScaler
```

> 🐍 `train_test_split(..., train_size=0.8)`, `StandardScaler().fit_transform`, `MinMaxScaler()`.

## 9.5 K-ближайших соседей

```lisp
;; x — точка, train-data — список векторов, train-labels — список меток
(defvar pred (knn-predict x train-data train-labels 3))
```

Реализация: считает евклидовы расстояния до всех точек, сортирует индексы (`argsort` через insert-sort), голосование по k соседям.

> 🐍 `KNeighborsClassifier(n_neighbors=3).fit(...).predict([x])`.

## 9.6 Наивный Байес (гауссов)

```lisp
(defvar model (nb-fit train-data train-labels))   ;; (classes priors stats)
(defvar pred (nb-predict x model))
```

> 🐍 `GaussianNB().fit(X, y).predict([x])`.

## 9.7 Деревья решений и ансамбли

```lisp
;; дерево: жадный поиск сплита по критерию Джини
(defvar tree (build-tree data labels 5))          ;; max-depth=5
(defvar pred (tree-predict tree x))

;; случайный лес: бутстрэп + голосование
(defvar forest (random-forest-fit data labels 10 5))
(defvar pred (random-forest-predict forest x))

;; градиентный бустинг: пни (stumps) на остатках
(defvar model (gb-fit data labels 50 0.1 3))      ;; n_estimators, lr, depth
(defvar pred (gb-predict model x))
```

> 🐍 `DecisionTreeClassifier(max_depth=5)`, `RandomForestClassifier(n_estimators=10)`, `GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)`.

## 9.8 K-Means

```lisp
(defvar centroids (kmeans-fit data 3 100))   ;; k=3, до 100 итераций
;; отнести точку к кластеру:
(defvar cluster (nearest-centroid x centroids 0 0 -1))
```

> 🐍 `KMeans(n_clusters=3, max_iter=100).fit(data).cluster_centers_`.

## 9.9 Кросс-валидация и подбор гиперпараметров

```lisp
;; k-fold CV: model-fn(train-x train-y) → модель с вызовом (model x)
(defvar score (cross-val-score X y 5 fit-fn score-fn))

;; перебор параметров
(defvar best (grid-search X y param-list fit-fn score-fn))
```

> 🐍 `cross_val_score(model, X, y, cv=5)` и `GridSearchCV(...)`.

## 9.10 End-to-end пример

```lisp
(import ML)

;; датасет: 2 класса, 2 признака (данные — списки)
(defvar data (list (list 1.0 2.0) (list 1.5 1.8) (list 5.0 8.0)
                   (list 6.0 9.0) (list 1.2 0.6) (list 5.5 8.2)))
(defvar labels (list 'a 'a 'b 'b 'a 'b))

;; масштабирование
(defvar X (map (\ (row) (standard-scale-lst row)) nil))  ;; по столбцам — упр. читателю
(defvar y labels)

;; KNN с кросс-валидацией (k=3)
(defun fit-knn (tx ty)
  (lambda (x) (knn-predict x tx ty 3)))

(defvar cv (cross-val-score data y 3 fit-knn accuracy-score))
(print cv)
```

> 🐍 Эквивалент:
> ```python
> from sklearn.neighbors import KNeighborsClassifier
> from sklearn.model_selection import cross_val_score
> model = KNeighborsClassifier(n_neighbors=3)
> cross_val_score(model, X, y, cv=3).mean()
> ```

## 9.11 Отличия от scikit-learn

- API функциональный: `(nb-fit ...)` вместо `.fit()`/`.predict()` классов.
- Данные — списки QLISP (для табличных данных), тензоры — для нейросетей.
- Реализации учебно-минималистичные: глядя в `stdlib/ml.qlsp`, вы видите каждый алгоритм целиком — это одновременно и документация, и курс по ML-алгоритмам.
