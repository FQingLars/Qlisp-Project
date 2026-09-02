# Глава 18. Шпаргалка: QLISP ↔ Python

Быстрый перевод между QLISP и Python-стеком (NumPy / PyTorch / scikit-learn). Имена примитивов в QLISP после `SymbolTable::intern` хранятся в верхнем регистре (`T+`, `MATMUL!`, `MSE!`, `PARAM`, `GRAD!`); пользователь может писать их в любом регистре.

## 18.1 Базовый синтаксис

| QLISP | Python |
|---|---|
| `(+ 1 2)` | `1 + 2` |
| `(f x y)` | `f(x, y)` |
| `(defun f (x) body)` | `def f(x): return body` |
| `(\ (x) (* x x))` | `lambda x: x * x` |
| `(defvar x 1)` | `x = 1` (модуль) |
| `(setq x 2)` | `x = 2` |
| `(let ((x 1) (y 2)) ...)` | `x, y = 1, 2` (в области) |
| `(if c a b)` | `a if c else b` |
| `(cond (c1 r1) (c2 r2) (t r))` | `if/elif/else` |
| `(while c body)` | `while c:` |
| `(progn a b c)` | тело функции / блок |
| `; комментарий` | `# комментарий` |
| `'(1 2 3)` | `[1, 2, 3]` (литерал) |
| `(equal a b)` | `a == b` |
| `(print "x" 42)` | `print("x", 42)` |
| `(type-of x)` | `type(x).__name__` |

## 18.2 Списки

| QLISP | Python |
|---|---|
| `(car xs)` / `(cdr xs)` | `xs[0]` / `xs[1:]` |
| `(cons x xs)` | `[x] + xs` |
| `(nth i xs)` | `xs[i]` |
| `(length xs)` | `len(xs)` |
| `(map f xs)` | `list(map(f, xs))` |
| `(filter p xs)` | `list(filter(p, xs))` |
| `(reduce + xs)` | `functools.reduce(op.add, xs)` |
| `(zip-with f xs ys)` | `[f(a,b) for a,b in zip(xs,ys)]` |
| `(member x xs)` | `x in xs` |
| `(range 5)` | `range(5)` / `list(range(5))` |
| `(take n xs)` / `(drop n xs)` | `xs[:n]` / `xs[n:]` |
| `(reverse xs)` | `xs[::-1]` |
| `(append xs ys)` | `xs + ys` |

## 18.3 Тензоры (NumPy)

| QLISP | NumPy |
|---|---|
| `(tensor ((1.0 2.0) (3.0 4.0)))` | `np.array([[1.,2.],[3.,4.]])` |
| `(zeros (list 2 3))` | `np.zeros((2,3))` |
| `(ones (list 4 2))` | `np.ones((4,2))` |
| `(randn (list 100 1))` | `np.random.randn(100,1)` |
| `(t+ a b)` / `(t* a b)` | `a + b` / `a * b` |
| `(matmul a b)` | `a @ b` |
| `(reshape t (list 4 1))` | `t.reshape(4,1)` |
| `(transpose t)` | `t.T` |
| `(tsum t)` / `(tsum t (list 0))` | `t.sum()` / `t.sum(axis=0)` |
| `(tmean t)` | `t.mean()` |
| `(argmax t)` | `t.argmax()` |
| `(tensor-shape t)` | `t.shape` |
| `(tensor-dtype t)` | `t.dtype` (`F32`/`F16`/`F64`/`I32`/`I64`/`U8`) |
| `(tsin t)` / `(tcos t)` / `(t/tanh t)` | `np.sin(t)` / `np.cos(t)` / `np.tanh(t)` |
| `(concat a b)` / `(stack a b)` | `np.concatenate([a,b])` / `np.stack([a,b])` |
| `(tensor-f16 t)` / `(tensor-f32 t)` / `(tensor-u8 t)` | `t.astype(np.float16/float32/uint8)` |
| `(tref t i j)` | `t[i, j]` |
| `(tensor-item t)` / `(tensor->list t)` | `t.reshape(-1)[0]` / `t.ravel().tolist()` |
| `(save-npy t "f.npy")` / `(load-npy "f.npy")` | `np.save` / `np.load` |

## 18.4 Автоград (PyTorch)

| QLISP | PyTorch |
|---|---|
| `(param (randn (list 4 2)))` | `torch.randn(4,2, requires_grad=True)` |
| `(matmul! x w)` | `x @ w` (в графе) |
| `(relu! h)` | `F.relu(h)` |
| `(mse! pred y)` | `F.mse_loss(pred, y)` |
| `(cross-entropy! logits y)` | `F.cross_entropy(logits, y)` |
| `(grad! loss)` | `loss.backward()` |
| `(grad-of w)` | `w.grad` |
| `(sgd-step w lr)` | SGD `opt.step()` (на одном парам.) |
| `(adam-step w lr)` | Adam `opt.step()` |
| обычные `matmul`, `t+` (без `!`) | `with torch.no_grad():` |

## 18.5 Слои (DL ↔ torch.nn)

| QLISP | PyTorch |
|---|---|
| `(linear 4 2)` (из DL) | `nn.Linear(4, 2)` |
| `(sequential (list l1 l2))` (из DL) | `nn.Sequential(l1, l2)` |
| `(layer x)` | `model(x)` / `forward` |
| `(batchnorm! h)` / `(layernorm! h)` | `nn.BatchNorm1d` / `nn.LayerNorm` |
| `(dropout! h 0.5)` | `nn.Dropout(0.5)` |
| `(conv2d! x k s p)` | `F.conv2d(x, k, stride=s, padding=p)` |
| `(maxpool2d! x k s)` | `F.max_pool2d(x, k, s)` |
| `(weighted-lookup probs v1 v2)` | взвешенная сумма (скаляр) |

## 18.6 Классический ML (sklearn)

| QLISP | scikit-learn |
|---|---|
| `(range 5)` / `(range 5 10 2)` | `range(5)` / `range(5, 10, 2)` |
| `(train-test-split data target ratio)` | `train_test_split(train_size=ratio)` |
| `(standard-scale-lst xs)` | `StandardScaler` |
| `(minmax-scale-lst xs)` | `MinMaxScaler` |
| `(knn-predict x X y k)` | `KNeighborsClassifier(k)` |
| `(nb-fit X y)` / `(nb-predict x m)` | `GaussianNB` |
| `(build-tree X y depth)` | `DecisionTreeClassifier(max_depth=...)` |
| `(random-forest-fit X y n d)` | `RandomForestClassifier(n, max_depth=d)` |
| `(gb-fit X y n lr d)` | `GradientBoostingClassifier(n, lr, d)` |
| `(kmeans-fit X k iters)` | `KMeans(k, max_iter=iters)` |
| `(accuracy-score yp yt)` | `accuracy_score` |
| `(cross-val-score X y k fit score)` | `cross_val_score(cv=k)` |
| `(grid-search X y params fit score)` | `GridSearchCV` |
| `(linear-kernel a b)` / `(rbf-kernel a b g)` | kernels из `sklearn.metrics.pairwise` |
| `(sigmoid-kernel a b alpha c)` | `tanh(α·⟨a,b⟩+c)` |

## 18.7 Символьный и нейросимвольный слой

| QLISP | Python-аналог |
|---|---|
| `(match v (p1 r1) (p2 r2) (_ def))` | `match v: case p1: r1; case p2: r2; case _: def` |
| `(unify a b)` | `unification.unify(a, b)` |
| `(put 'cat 'legs 4)` / `(get 'cat 'legs)` | атрибуты у enum'ов / `obj.attr = 4` |
| `(assert-fact (cat ?x))` + `(defrule ...)` + `(run-rules)` + `(query ...)` | pyDatalog / CLIPS / Prolog |
| `(amb-let ...)` + `(require ...)` + `(amb-collect ...)` | `itertools.product` + backtracking |
| `(ns-if logits (("a") branch-a) (("b") branch-b))` | tree-of-experts (нет аналога) |
| `(ns-grad! loss 'cat)` | per-sample STE + guilt detector (нет аналога) |

## 18.8 Строки, файлы, прочее

| QLISP | Python |
|---|---|
| `(strcat a b)` | `a + b` / `f"{a}{b}"` |
| `(substr s i n)` | `s[i:i+n]` |
| `(number-to-string 42)` | `str(42)` |
| `(slurp "f")` / `(spit "f" s)` | `open("f").read()` / `.write(s)` |
| `(read-lines "f")` | `open("f").readlines()` |
| `(read-csv "f.csv")` | `csv.reader(open("f.csv"))` |
| `(re-find p s)` / `(re-replace p r s)` | `re.search` / `re.sub` |
| `(re-split p s)` | `re.split(p, s)` |
| `(now-ts)` / `(sleep s)` | `time.time()` / `time.sleep(s)` |
| `(now-iso-str)` | `datetime.now().isoformat()` |
| `(elapsed expr)` | contextmanager `timeit`/декоратор |
| `(http-get url)` | `requests.get(url).text` |
| `(dashboard port)` + `(log-metric "loss" v)` | TensorBoard |

## 18.9 Компиляция

| QLISP | Python-мир |
|---|---|
| `qlispc prog.qlsp -o prog` | `nuitka --onefile prog.py` |
| `(start-trace)` + `(graph-param x)` + `(hlo-compile out)` + `(hlo-run hlo x ...)` | `torch.compile` / `jax.jit` |
| `(defuse! f (x) ...)` | `@torch.compile(fullgraph=True)` |
| `(defuse f (x) ...)` | `@torch.compile` (с fallback) |
| `QLISP_TARGET_TRIPLE=aarch64-linux-gnu` | кросс-компиляция под ARM |

## 18.10 FFI

| QLISP | Python-мир |
|---|---|
| `(ffi-load "lib.so")` | `ctypes.CDLL("lib.so")` |
| `(ffi-call "sin" "float" 0.0)` | `lib.sin(0.0)` |
| `(ffi-call "atan2" "float" 1.0 1.0)` | `lib.atan2(1.0, 1.0)` |
| `(ffi-call "strlen" "int" "hello")` | `lib.strlen(b"hello")` |
| `(ffi-import "libfoo.so" 'fn)` | `cdll.libfoo.fn` |