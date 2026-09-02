# Глава 14. Символьное и нейросимвольное программирование

QLISP — не просто «Лисп для тензоров»: в ядро встроен **символьный движок** (S1–S8) и поверх — **нейросимвольный маршрутизатор** (NS-IF/NS-GRAD! с guilt detector, S9). Это делает QLISP подходящим для expert systems, knowledge graphs и нейросимвольного AI.

Реализация — `src/symbolic/engine.cpp` (специальные формы S1–S8), `src/ns/route.cpp` (нейросимвольный роутер) и встроенные примитивы (`MATCH`, `UNIFY`, `SUBST`, `DEFRULE`, `ASSERT-FACT`, `RUN-RULES`, `QUERY`, `AMB-LET`, `REQUIRE`, `AMB-COLLECT`, `DEFINE-COMPILER-MACRO`, `DEFHLOOP`, `NS-IF`, `NS-GRAD!`).

## 14.1 Pattern matching: `match`

Специальная форма `MATCH` → `Interpreter::do_match`:

```lisp
(match value
  (pattern-1 результат-1)
  (pattern-2 результат-2)
  (_ дефолт))
```

Паттерны (см. `match_pattern` в `engine.cpp`):

| Паттерн | Что сопоставляет |
|---|---|
| `_` | что угодно (wildcard) |
| `?x` или `(? x)` | что угодно, связывает переменную `x` |
| `42`, `3.14`, `"hi"`, `nil`, `'sym` | литерал |
| `(quote val)` | литерал (явно через `quote`) |
| `(list a b)` | список фиксированной длины |
| `(cons h t)` | cons-ячейка: голова + хвост |
| `(or p1 p2)` | любой из паттернов |
| `(when p guard)` | паттерн + предикат-guard |
| `(quote x)` | литерал |

Примеры:

```lisp
(match 42
  (42 "сорок два")
  (_ "другое"))                       ;; → "сорок два"

(match (list 1 2 3)
  ((list ?a ?b ?c) (+ ?a ?b ?c))
  (_ 0))                              ;; → 6

(match (cons 1 2)
  ((cons ?a ?b) (+ ?a ?b)))           ;; → 3

(match (list 1 2 3)
  ((list ?x ?y ?z) (when (> ?x 0)) (+ ?y ?z))
  (_ 0))                              ;; → 5
```

> 🐍 **Python-аналогия.** Это `match/case` из Python 3.10+, но с настоящими биндингами переменных по структуре:
> ```python
> match value:
>     case [a, b, c]: return a + b + c
>     case _: return 0
> ```

Если ни один паттерн не подошёл — `match` бросает `runtime_error` (`throw std::runtime_error("match: no pattern matched")`).

## 14.2 Унификация: `unify` и `subst`

Унификация — сердце Prolog: найти подстановку переменных, при которой два выражения совпадут (с проверкой occurs check):

```lisp
(unify '(?x) '(foo bar))             ;; → ((?X . (FOO BAR)))
(unify '(f (?x) b) '(f a b))         ;; → ((?X . A))
(subst '((?x) . replacement) expression)   ;; применить подстановку
```

Логические переменные — символы, начинающиеся с `?` (`is_lvar` в `engine.cpp:79`): `?x`, или cons-форма `(? x)`.

Реализация (`engine.cpp:78-112`): структурная унификация через `walk`/`occurs`/`extend`/`unify_terms`. `occurs` предотвращает бесконечные циклы (X = f(X) недопустимо).

> 🐍 В Python унификацию делают библиотеки типа `unification`/`kanren` — в QLISP она встроена.

## 14.3 Property lists: свойства символов

Каждый символ может нести словарь свойств (глобально, в stable-памяти). Реализация — `register_plist_primitives` (`engine.cpp:116-145`): `plist_map` — `unordered_map<Symbol*, unordered_map<Symbol*, SExpr*>>`. При `PUT` значение клонируется в `stable_scope()`.

```lisp
(put 'cat 'legs 4)
(get 'cat 'legs)        ;; → 4
(symbol-plist 'cat)     ;; все свойства
(remprop 'cat 'legs)
```

Оптимизация: для `fixnum`/`flonum`/`symbol`-значений PUT **обновляет in-place** (без нового конс-ячейки).

> 🐍 Это «атрибуты у enum'ов»: `class Animal(Enum): CAT = ...` + `CAT.legs = 4`. Классика Common Lisp, удобна для метаданных.

## 14.4 Rule engine: факты и правила

Forward-chaining движок. Факты хранятся в `kb_facts` (`vector<SExpr*>`, элементы `clone_to(root_arena_)` для immortality). Правила — в `rules_db` (`vector<Rule>`), где `Rule { name, whens, asserts }`.

```lisp
;; факты (цитируются — это данные)
(assert-fact (warm-blooded? cat))
(assert-fact (has-fur? cat))
(assert-fact (live-birth? cat))

;; правило: все тёплые+шерсть+живорождение — млекопитающие
;; Синтаксис 1: (WHEN ...) (ASSERT ...) clauses
(defrule mammal ((? x))
  (when (warm-blooded? (? x))
        (has-fur? (? x))
        (live-birth? (? x)))
  (assert (mammal? (? x))))

;; Синтаксис 2: через =>
(defrule mammal2 ((? x))
  (warm-blooded? (? x))
  (has-fur? (? x))
  =>
  (mammal? (? x)))

(run-rules)                    ;; прогнать вывод (до 100 итераций по умолчанию)

(query (mammal? cat))          ;; → подстановки (здесь (((?X . CAT))))
(query (mammal? (? who)))      ;; → все решения переменной who
(retract-fact (has-fur? cat))  ;; убрать факт
```

Реализация (`do_defrule`/`do_run_rules`/`do_query`):

- Каждое правило парсится: `(when cond1 cond2 ...)` → `whens`; `(assert fact1 fact2 ...)` или после `=>` — `asserts`. Оба списка клонируются в root (immortal).
- `find_bindings(whens)` перебирает все факты в KB, унифицирует whens-цепочку с фактами, собирает все успешные подстановки.
- `run-rules` итерирует до фиксации (`max=100` итераций) или пока добавляются новые факты.
- `query pattern` унифицирует pattern со всеми фактами, возвращает список подстановок.

Правила могут **цепляться** (выведенный факт запускает следующие правила).

> 🐍 Прямого аналога в Python нет — это territory pyDatalog/CLIPS/Prolog. Но обратите внимание: факты и правила — обычные S-выражения, их можно строить макросами (гл. 12).

## 14.5 Недетерминизм: `amb-let`, `require`, `amb-collect`

Логическое программирование через «amb» (ambiguous choice) с бэктрекингом (через `AmbFail` exception):

```lisp
(amb-collect
  (amb-let (x '(1 2 3))
    (amb-let (y '(1 2 3))
      (require (> (* x y) 4))
      (list x y))))
;; все пары (x y), где x*y > 4
```

Реализация (`do_amb_let`/`do_require`/`do_amb_collect`):

- `amb-let` создаёт декартово произведение значений переменных и перебирает индексы (`std::vector<size_t>`), пока `body` не упадёт в `AmbFail`.
- `require` бросает `AmbFail` если условие ложно → переход к следующей комбинации.
- `amb-collect` собирает все успешные результаты в список.

> 🐍 Аналог — `itertools.product` + фильтр: `[(x, y) for x in xs for y in ys if x*y > 4]`.

## 14.6 Reader macros

`SET-READER-MACRO` (примитив `prim_set_reader_macro` в `interpreter.cpp:1690`) — регистрация пользовательских литералов на уровне ридера.

## 14.7 Compiler macros: оптимизация на этапе компиляции

Специальная форма `DEFINE-COMPILER-MACRO` → `do_defcompiler_macro`. Макрос сохраняется в `MacroExpander::compiler_macros_` (через `defcompiler_macro` в `expander.cpp:66`). Тело и параметры клонируются в root (immortal).

```lisp
(define-compiler-macro square (x)
  (if (constantp x)
      (* x x)
      `(* ,x ,x)))
```

Это нужно для оптимизации символьных выражений при AOT-компиляции.

## 14.8 Нейросимвольный слой: `NS-IF` и `NS-GRAD!`

Уникальная конструкция QLISP: **нейросимвольное дерево**. Символьная развилка `NS-IF` маршрутизирует данные в одну из нейросетевых ветвей, и автоградиент (гл. 7) корректно распределяет ошибку через `NS-GRAD!`.

Реализация — `src/ns/route.{hpp,cpp}` (классы `RouteHop`, `Router`, `GuiltDetector`, `GuiltVerdict`).

```lisp
(defvar logits (matmul! x router-w))    ;; логиты решения

(defvar pred (ns-if logits
  (("dog") (matmul! x dog-net))         ;; ветка «собака»
  (("cat") (matmul! x cat-net))))       ;; ветка «кошка»

(defvar loss (mse! pred y-true))
(ns-grad! loss 'cat)                    ;; backward с учётом маршрута
```

Специальная форма `NS-IF` → `Interpreter::do_ns_if` (`interpreter.cpp:694`):

1. Принимает логиты `[N]` (один образец) или `[batch, N]` (батч).
2. Каждая ветвь задаётся `(("label1" "label2") body...)` — `branch_covers` описывает, какие ground-truth-метки покрывает ветвь.
3. Выбирает ветвь per-sample по argmax, записывает `RouteHop` в `Router`, возвращает результат **первой выбранной ветви** (для чистого батча корректно; для смешанного — маршрут/guilt верны).
4. Градиент через развилку не течёт автоматически — лента **разорвана на точке выбора** (см. TODO §1.1: в планах — ROUTE-узел в HLO).

`NS-GRAD!` → `do_ns_grad` (`interpreter.cpp:768`):

1. Принимает loss и label (символ/строка или список).
2. Вызывает `GuiltDetector::detect`, который для каждого образца находит **первый (ближайший к корню) хоп**, где `taken != correct` (`RouteHop::resolve_correct`).
3. Если расхождений нет → `GuiltKind::NONE` (виновен лист); иначе `GuiltKind::ROUTING` (виновен маршрутизатор).
4. Per-sample softmax-weighted STE в `decision_logits->grad`:
   - виновный образец: `+(1 - p_correct)` к правильной ветви, `-p_taken` к взятой.
5. **Блокировка** tape-узлов взятой ветви: целиком, если ВСЕ образцы в нём виновны; при смешанном — частично (правильные сохраняют градиент).
6. Затем `GradientTape::backward(loss, arena, blocked)` идёт в reverse-topo с пропуском заблокированных.

### Что «виновно»

| Случай | Виновник | Градиент |
|---|---|---|
| Маршрут верный | лист | в листовую ветвь (нормальный backward) |
| Развилка неверна (per sample) | маршрутизатор | per-sample STE в логиты развилки; виновные ветви блокируются |
| Глубокое дерево | первый расходящийся узел | STE туда; подветви блокируются |
| Смешанный батч | по образцу | STE/блокировка per-sample; правильные сохраняют градиент |
| OR-overlap (несколько покрывающих ветвей) | taken-ветвь первая | если она покрывает label → вердикт NONE (виновен лист) |

### Почему это уникально

> 🐍 Это гибрид `nn.Module` с `torch.gather`-маршрутизацией и hard attention: представьте дерево if-ов, где каждая развилка — обучаемая нейросеть, и весь граф дифференцируем. В PyTorch такого из коробки нет — здесь это две формы языка.

## 14.9 Когда символьный слой полезен

- Expert systems поверх ML-моделей: правила фильтруют/маршрутизируют предсказания (`NS-IF` маршрутизирует в ветвь, обученную под конкретный класс; правила могут менять label-set).
- Knowledge representation: факты + унификация + запросы без внешнего Prolog.
- Обучаемые решающие деревья с нейросетями в листьях — `NS-IF` ровно про это.

Символьный слой живёт на S-выражениях в stable-памяти (гомоиконность), нейронный — компилируется через HLO (гл. 10). QLISP сознательно разделяет эти миры: интроспекция правил не мешает нативной скорости тензоров.

## 14.10 Известные ограничения

- `unify`: нет backtracking (S5 в `AMB-LET` компенсирует, но не для самой `unify`).
- `match`: нет type-check паттернов (`(tensor ?x)` не работает — будет интерпретировано как cons-литерал).
- `NS-IF`: для смешанного батча исполняется ветвь первого образца (маршрут/guilt per-sample верны, но forward-вычисление — только по взятому маршруту). В планах — ROUTE-узел в HLO-графе с per-sample-исполнением (`TODO.md` §1.1).
- `NS-IF`: глобальная вина «по причине» (какое правило произвело спорный факт) требует полноценного rule engine в NS-движке — пока resolve_correct лишь OR-aware, не causal.