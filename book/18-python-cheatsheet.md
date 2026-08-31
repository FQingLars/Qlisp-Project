# Глава 18. Шпаргалка: QLISP ↔ Python

Быстрый перевод между QLISP и Python-стеком (NumPy / PyTorch / scikit-learn).

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
| `(sqrt t)` / `(exp t)` / `(log t)` | `np.sqrt(t)` / `np.exp(t)` / `np.log(t)` |
| `(stack ...)` / `(concat ...)` | `np.stack(...)` / `np.concatenate(...)` |
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
| `(linear 4 2)` | `nn.Linear(4, 2)` |
| `(sequential (list l1 l2))` | `nn.Sequential(l1, l2)` |
| `(layer x)` | `model(x)` / `forward` |
| `(batchnorm! h)` / `(layernorm! h)` | `nn.BatchNorm1d` / `nn.LayerNorm` |
| `(dropout! h 0.5)` | `nn.Dropout(0.5)` |
| `(conv2d! x k s p)` | `F.conv2d(x, k, stride=s, padding=p)` |
| `(maxpool2d! x k s)` | `F.max_pool2d(x, k, s)` |
| `(weighted-lookup idx W)` | `nn.Embedding` lookup |

## 18.6 Классический ML (sklearn)

| QLISP | scikit-learn |
|---|---|
| `(train-test-split data target 0.8)` | `train_test_split(train_size=0.8)` |
| `(standard-scale-lst xs)` | `StandardScaler` |
| `(minmax-scale-lst xs)` | `MinMaxScaler` |
| `(knn-predict x X y k)` | `KNeighborsClassifier(k)` |
| `(nb-fit X y)` / `(nb-predict x m)` | `GaussianNB` |
| `(build-tree X y depth)` | `DecisionTreeClassifier` |
| `(random-forest-fit X y n d)` | `RandomForestClassifier` |
| `(gb-fit X y n lr d)` | `GradientBoostingClassifier` |
| `(kmeans-fit X k iters)` | `KMeans` |
| `(accuracy-score yp yt)` | `accuracy_score` |
| `(cross-val-score X y k fit score)` | `cross_val_score(cv=k)` |
| `(grid-search X y params fit score)` | `GridSearchCV` |
| `(linear-kernel a b)` / `(rbf-kernel a b g)` | kernels из `sklearn.metrics.pairwise` |

## 18.7 Строки, файлы, прочее

| QLISP | Python |
|---|---|
| `(strcat a b)` | `a + b` / `f"{a}{b}"` |
| `(substr s i n)` | `s[i:i+n]` |
| `(number-to-string 42)` | `str(42)` |
| `(slurp "f")` / `(spit "f" s)` | `open("f").read()` / `.write(s)` |
| `(read-lines "f")` | `open("f").readlines()` |
| `(re-find p s)` / `(re-replace p r s)` | `re.search` / `re.sub` |
| `(now-ts)` / `(sleep s)` | `time.time()` / `time.sleep` |
| `(json-parse s)` / `(json-stringify d)` | `json.loads` / `json.dumps` |
| `(ffi-load "lib.so")` | `ctypes.CDLL` |
| `(http-get url)` | `requests.get(url).text` |
| `(type-of x)` | `type(x).__name__` |

## 18.8 Компиляция

| QLISP | Python-мир |
|---|---|
| `qlispc prog.qlsp -o prog` | `nuitka --onefile prog.py` |
| `(START-TRACE)` + `(HLO-COMPILE out)` + `(HLO-RUN ...)` | `torch.compile` / `jax.jit` |
| `(defuse! f (x) ...)` | `@torch.compile(fullgraph=True)` |
| `(defuse f (x) ...)` | `@torch.compile` (с fallback) |
| `QLISP_TARGET_TRIPLE=aarch64-linux-gnu` | кросс-компиляция под ARM |
