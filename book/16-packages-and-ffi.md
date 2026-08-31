# Глава 16. Пакеты и FFI

## 16.1 Два способа ставить пакеты

| Способ | Что это | Python-аналог |
|---|---|---|
| `qvalent` (CLI) | Полноценный пакетный менеджер с манифестом `qvalent.toml` | `pip` + `pyproject.toml` |
| `(pkg-install "user/repo")` | Лёгкая установка из stdlib (один файл `pkg.qlsp` из GitHub) | `curl install.sh` |

## 16.2 qvalent: pip для QLISP

Пакет — репозиторий GitHub с манифестом `qvalent.toml`:

```toml
[package]
name = "my-package"
version = "0.1.0"
description = "A QLISP package"
deployer = "your-github-username"
repo = "repo-name"

[dependencies]
```

Команды:

```bash
qvalent init                          # создать qvalent.toml
qvalent get user/repo@v1.0.0          # установить (версия опциональна)
qvalent list                          # установленные пакеты
qvalent info my-package               # детали пакета
qvalent update my-package             # обновить
qvalent remove my-package             # удалить
qvalent search tensors                # поиск по GitHub (topic:qlisp)
```

Пакеты скачиваются tarball'ом из GitHub API, кэшируются и ставятся в локальный каталог пакетов.

> 🐍 `qvalent get user/repo` ≈ `pip install git+https://github.com/user/repo`; `qvalent search` ≈ `pip search` (который в pip давно отключили).

## 16.3 pkg-install: минимум

Модуль `PKG` качает `pkg.qlsp` из ветки `main` репозитория и перечисленные в нём файлы:

```lisp
(import PKG)
(pkg-install "user/repo")
```

Годится для простых библиотек-однофайловок; для всего остального — `qvalent`.

## 16.4 FFI: вызов C-библиотек

QLISP умеет загружать разделяемые библиотеки и вызывать C-функции напрямую:

```lisp
(defvar libm (ffi-load "libm.so.6"))               ;; dlopen
(ffi-call libm "sin" 0.0)                          ;; вызвать C-функцию
(ffi-import libm "cos" 'float 'float)              ;; обёртка: Lisp-функция
```

| QLISP-тип | C-тип |
|---|---|
| fixnum | `int` / `int64_t` |
| flonum | `float` / `double` |
| string | `const char*` |
| tensor (U8) | `uint8_t*` |
| tensor (F32) | `float*` |

> 🐍 **Python-аналогия.** Это `ctypes`:
> ```python
> import ctypes
> libm = ctypes.CDLL("libm.so.6")
> libm.sin(0.0)
> ```

Так подключаются OpenBLAS, CUDA-библиотеки, любые C/C++ SDK — без написания биндингов.

## 16.5 Обмен данными с Python-экосистемой

Прямой мост (без FFI) — форматы данных:

```lisp
;; NumPy
(save-npy tensor "w.npy")          ;; np.save
(defvar w (load-npy "w.npy"))      ;; np.load

;; JSON
(json-stringify data)              ;; json.dumps
(json-parse str)                   ;; json.loads

;; CSV
(csv-read "data.csv")
(csv-write rows "out.csv")
```

HTTP из коробки:

```lisp
(http-get "https://api.github.com/repos/FQingLars/QLISP")   ;; requests.get (GET only)
```

Файловая система:

```lisp
(glob "*.qlsp") (listdir ".") (mkdir "dir") (delete-file "f")
(path-exists? "p") (path-join "a" "b") (path-dirname "a/b") (path-basename "a/b")
(getenv "HOME")
```

## 16.6 Сериализация QSRD

Встроенный формат сериализации QLISP (qserde):

```lisp
(qsrd-save tensor "model.qsrd")
(defvar t (qsrd-load "model.qsrd"))
(qsrd-get ...)
```

Удобен для чекпойнтов моделей, когда не нужна совместимость с NumPy.
