# Глава 13. ООП

В QLISP две ООП-системы: лёгкие записи (`defstruct`) и полноценный CLOS-подобный MOP (`defclass`, дженерики).

## 13.1 defstruct: структуры-записи

```lisp
(defstruct Point (x y))

(defvar p (make-instance 'Point 'x 10 'y 20))
(slot-value p 'x)          ;; → 10
```

> 🐍 **Python-аналогия.** Это `@dataclass`:
> ```python
> @dataclass
> class Point:
>     x: int
>     y: int
> p = Point(x=10, y=20); p.x
> ```

`defstruct` автоматически создаёт конструктор и слоты; экземпляры — обычные значения (живут в арене, гл. 5).

## 13.2 defclass: классы со слотами

```lisp
(defclass Animal ()
  (name species))

(defvar cat (make-instance 'Animal 'name "Whiskers" 'species "cat"))
(slot-value cat 'name)     ;; → "Whiskers"
```

> 🐍 ≈ обычный `class Animal: __init__(self, name, species)`.

## 13.3 Дженерики: defgeneric + defmethod

Полиморфизм по классу аргумента — через обобщённые функции (как в CLOS, а не как в Java):

```lisp
(defgeneric area (shape))

(defmethod area ((shape Circle))    (* 3.14159 (slot-value shape 'r) (slot-value shape 'r)))
(defmethod area ((shape Rect))      (* (slot-value shape 'w) (slot-value shape 'h)))

(area circle1)    ;; диспетчеризация по типу объекта
(area rect1)
```

> 🐍 Python-аналог — `functools.singledispatch`:
> ```python
> @singledispatch
> def area(s): ...
> @area.register
> def _(s: Circle): return 3.14159 * s.r ** 2
> ```
> Разница: в QLISP метод — часть системы классов (MOP), диспетчеризация встроенная.

## 13.4 Функциональный стиль: слои вместо классов

Важно понимать идиоматику ML-кода QLISP: **модели обычно пишут не классами, а замыканиями** (гл. 8):

```lisp
(defvar model (linear 4 2))     ;; замыкание с параметрами внутри
(model x)
```

Классы уместны для **данных** (датасеты, конфиги, узлы деревьев), замыкания — для **поведения** (слои, модели). Это противоположно PyTorch, где `nn.Module` — класс.

## 13.5 Сводная таблица

| QLISP | Python | Комментарий |
|---|---|---|
| `(defstruct Point (x y))` | `@dataclass class Point:` | записи |
| `(defclass A () (slot1 slot2))` | `class A:` | классы |
| `(make-instance 'A 'slot v)` | `A(slot=v)` | конструктор |
| `(slot-value obj 'slot)` | `obj.slot` | доступ к полю |
| `(defgeneric f (x))` + `(defmethod f ((x A)))` | `singledispatch` | полиморфизм |
