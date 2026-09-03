# QLISP 2.3.0 — Release Notes

**Тег:** `v2.3.0`  
**Базовый релиз:** `v2.2.1` (11 коммитов изменений)  
**Платформы:** Linux x86_64, Windows x86_64 (MSYS2 MinGW64)

## Бинарники

| Файл | Платформа | Состав |
|------|-----------|--------|
| `releases/2.3.0/qlisp-2.3.0-linux-x86_64.tar.gz` | Linux x86_64 | `qlisp`, `qlispc`, `qlisp-lsp`, `qvalent` |
| GitHub Release `v2.3.0` → `qlisp-windows-x86_64.zip` | Windows x86_64 | `qlisp.exe`, `qlispc.exe`, `qlisp-lsp.exe`, `qvalent.exe` |
| GitHub Release `v2.3.0` → `qlisp-linux-x86_64` | Linux x86_64 | те же четыре бинарника (CI-сборка) |

Windows-бинарники собирает CI (`.github/workflows/build.yml`, job `windows`):
MSYS2 MinGW64, LLVM/OpenBLAS нативно под Win64 ABI. Публикация релиза
происходит автоматически при пуше тега `v2.3.0`
(`git tag v2.3.0 && git push origin main v2.3.0`).

> Примечание о локальной кросс-сборке: кросс-компиляция `qlisp`/`qlispc`
> под Windows из Linux линкуется против host-LLVM (rocm `libLLVM*.a`),
> собранного с SysV ABI — такие объектные файлы принципиально несовместимы
> с Win64 ABI. Кросс-путь в CMake оставлен только для `qlisp-lsp`/`qvalent`
> (без LLVM, работает: добавлены шимы заголовков `uid_t/gid_t/nlink_t`).
> Единственный корректный путь для компилятора — нативная сборка MSYS2
> или CI, что и используется в релизе.

## Языковые нововведения (с 2.2.1)

### 1. Гомоиконность: код как данные (`fae86d6`)

Новые примитивы:

| Примитив | Что делает |
|----------|------------|
| `READ-FROM-STRING` | парсит строку в S-выражение (читатель применяется в рантайме) |
| `WRITE-TO-STRING` | сериализует S-выражение обратно в строку |
| `EVAL` | вычисляет S-выражение (в т.ч. построенное в рантайме) |
| `FUNCTIONP` | предикат «значение — функция» |
| `FUNCTION-PARAMS` | список параметров лямбды |
| `FUNCTION-BODY` | тело лямбды как S-выражение |
| `FUNCTION-ENV` | окружение замыкания (инспекция) |
| `SYMBOL-NAME` | имя символа как строка |
| `SET-READER-MACRO` | пользовательский макрос чтения |
| `RANDINT` | случайное целое |

Пример: `(eval (read-from-string "(+ 1 2)"))` → `3`.

### 2. QSRD v2 — структурные блобы (`82969af`)

Бинарный формат данных получил версию 2 (magic `"QSRD"`, u16 ver): блоб типа
`2` — S-выражение в бинарной сериализации (структуры читаются/пишутся как
данные), тип `3` — типизированный тензор (dtype сохраняется вместе с шейпом).
Примитивы: `QSRD-SAVE`, `QSRD-LOAD`, `QSRD-GET`.

### 3. Модуль `string` (`87ab02a`)

Новый стандартный модуль, два слоя:

- Примитивы: `STRING-LENGTH`, `STRING-SPLIT`, `STRING-JOIN`, `STRING-UPCASE`,
  `STRING-DOWNCASE`, `STRING-TRIM`, `STRING-REPLACE`, `STRING-INDEX`,
  `STRING-REVERSE`, `CHAR-CODE`, `CODE-CHAR`, `STRING-TO-NUMBER`, `STRING`.
- Составные (на примитивах): `string-contains`, `string-starts-with`,
  `string-ends-with`, `string-slice`, `string-lines`, `string-words`,
  `string-repeat`, `to-string`.

Также в этом коммите: `type-of` возвращает uppercase-имена типов
(`INT`, `FLOAT`, `STRING`, `SYMBOL`, `LIST`, `TENSOR`, `FUNCTION`, …),
cons-ячейки корректно сообщаются как `LIST`.

### 4. `IMPORT-FROM`, файловые модули, `BOUNDP` (`c3401e1`)

- `IMPORT-FROM` — частичный импорт: `(import-from (dl linear train-step))`
  поднимает только перечисленные имена.
- Модули теперь могут жить в файлах рядом со скриптом (embed-таблица →
  файловый фолбэк).
- `BOUNDP` — предикат «имя связано» (разрешение без чтения значения).

### 5. Модуль `dl` переписан: слои как данные (`19dafbb`)

Легаси-`dl.qlsp` заменён целиком. Модель — обычный список-структура, слои —
данные, а не объекты:

```lisp
(defvar model (list (linear 2 8) 'relu (linear 8 1)))
(net-forward model x)                 ; прямой проход (рекурсия, без while)
(train-step model optimizer x y lr)   ; forward + MSE! + GRAD! + шаг оптимизатора
(train-epochs model optimizer xs ys lr epochs)
```

- Слой: `(list 'linear W b)`, где `W`/`b` созданы `param` (requires_grad).
- Активации — голые символы `'relu` / `'sigmoid` / `'tanh` / `'softmax`.
- Оптимизаторы: `'sgd` / `'adam` (символы), шаги — `SGD-STEP`/`ADAM-STEP`.
- Внутри — только операции с tape-контрактом (`MATMUL!`, `T+!`, `MSE!`, …).

## Исправления

### Gradient tape переживает границы форм (TBC #26, `371dd46`)

Канонический паттерн «граф в одной форме, `GRAD!` в следующей» раньше
опирался на то, что память нод tape ещё не переиспользована (UB: флаки,
зависания, `heap-use-after-free` под ASan). Теперь:

- ноды tape принадлежат самой tape (heap, удаляются в `clear()`);
- promote тензора (`defvar`/`setq`) перенаправляет ребро `result` ноды
  на stable-копию — градиент течёт в тензор, который видит пользователь;
- `reset_scratch` «запечатывает» незакрытые графы: scratch-тензоры,
  на которые ссылаются живые ноды, клонируются в stable (один раз на
  тензор), рёбра переписываются. Обучающие циклы ничего не платят
  (tape пуст после каждого `GRAD!`);
- `defvar_autograd`, `hlo_training`, `hlo_codegen`, `dl` — детерминированно
  зелёные в Release и ASan;
- HLO/NS-прокси стали строковыми хендлами: биндинги копируют скаляры,
  и fixnum-прокси терял идентичность (ключ `trace_nodes_`).

### Биндинги не алиасят литералы; Adam владеет своим состоянием (`f61426b`)

- `let`/`bind_and_eval` копируют скалярные значения (fixnum/flonum/symbol)
  в свежие ячейки — `setq` больше не мутирует исходные литералы в коде.
- `ADAM-STEP` аллоцирует `m`/`v` состояния параметра в stable-scope —
  состояние оптимизатора не умирает с итерацией.

### HLO in-process AOT (`ef4994e`)

Компиляция HLO-кернелов больше не вызывает `popen`+`llc`: LLVM-эмит и
линковка LLD происходят в процессе. Для Linux-линковки используется
in-process `lldELF` (детект через CMake, `QLISP_HAS_LLD`). Пайплайн:
`HloGraph → LLVM emit → object → lld → .so → dlopen`.

### Сборка (`1bf13b4`)

- `find_package(LLVM)` запинен к установке, на которую указывает
  `llvm-config` (исключён ABI-микс заголовков/библиотек из разных LLVM).
- `run_file` сбрасывает scratch после каждой формы верхнего уровня
  (разделение lifetime форм в скриптах).
- In-process LLD требует линковки `lld` + `zlib` + `zstd` (Debian-style
  линкеры требуют явный `-lz -lzstd`; CI обновлён: `lld-22 liblld-22-dev`).

## Контракт gradient tape (для авторов модулей)

- Градиенты считаются только через `!`-операции: `MATMUL!`, `T+!`, `T*!`,
  `MSE!`, `RELU!`, `SIGMOID!`, `TANH!`, `SOFTMAX!`, `LOG!`, `EXP!`, `TSIN!`,
  `TCOS!`, `CONV2D!`, `MAXPOOL2D!`, `BATCHNORM!`, `LAYERNORM!`, `DROPOUT!`,
  `CROSS-ENTROPY!`. Обратной разницы «обычных» операций нет (`T-!` не существует).
- `GRAD!` = backward + очистка tape.
- `GRAD-OF` читается в той же форме, где был `GRAD!` (тензоры градиентов
  живут в scratch текущей формы).
- Промежуточные тензоры не стоит держать через `defvar` (промоут клонирует
  тензор); алиасы параметров — через `let`.

## Тесты

Сюит: **48/50** (Release, Linux). Два известных детерминированных фейла,
оба существовали до 2.3.0 и вне скоупа релиза:

- `tests/ns_router.qlsp` → `ns-nested-breed-guilty` (guilt-блокировка
  в вложенном NS-IF);
- `tests/symbolic_neuro.qlsp` → `neurosym-rule-check` (символический
  движок правил).

ASan: чисто на tape-тяжёлых тестах (`dl`, `defvar_autograd`,
`hlo_training`) — `heap-use-after-free` из 2.2.x не воспроизводится.

## Известные ограничения

- HLO fallback-граф (без `HLO-COMPILE`) держит константы формы трассировки
  — кросс-форменное использование такого HLOPROG небезопасно (TEMP_ISSUES #23).
- Shell-линковка HLO-кернела — fallback при сборке без LLD.
- Windows: локальная кросс-сборка компилятора из Linux невозможна (ABI);
  используйте CI-релиз или MSYS2 (`README.md`, раздел Windows).

## Обновление гайда языка

Для соответствия гайда 2.3.0 нужно дополнить разделы: гомоиконные
примитивы (п.1), бинарный формат QSRD v2 (п.2), модуль `string` (п.3),
`IMPORT-FROM`/`BOUNDP` (п.4), модуль `dl` (п.5), контракт tape
(раздел выше).
