# Глава 16. Пакеты и FFI

## 16.1 Два способа ставить пакеты

| Способ | Что это | Python-аналог |
|---|---|---|
| `qvalent` (CLI) | Полноценный пакетный менеджер с манифестом `qvalent.toml` | `pip` + `pyproject.toml` |
| `(pkg-install "user/repo")` | Лёгкая установка из stdlib (через `http-get` примитив) | `curl install.sh` |

## 16.2 qvalent: pip для QLISP

CLI `qvalent` (`src/qvalent/main.cpp`, ~260 строк) — отдельная цель CMake. Команды (из `print_usage()`):

```bash
qvalent init                                 # создать qvalent.toml в текущей директории
qvalent get user/repo[@version]              # установить (версия опциональна)
qvalent list                                 # установленные пакеты
qvalent info <name>                          # детали пакета
qvalent update <name>                        # обновить
qvalent remove <name>                        # удалить
qvalent search <query>                       # поиск по GitHub (topic:qlisp)
qvalent help                                 # эта справка
```

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

Реализация: парсер TOML — крошечный встроенный (`parse_qvalent_toml` в `main.cpp:46-70`). Скачивание — через `curl -L` tarball/zip с GitHub API → `$HOME/.cache/qlisp/<deployer>/<repo>/` → распаковка в `qlisp-packages/<name>/`.

> 🐍 `qvalent get user/repo` ≈ `pip install git+https://github.com/user/repo`; `qvalent search` ≈ `pip search` (который в pip давно отключили).

## 16.3 pkg-install: минимум

Модуль `PKG` (`stdlib/pkg.qlsp`) — качает `pkg.qlsp` из ветки `main` репозитория и перечисленные в нём файлы:

```lisp
(import PKG)
(pkg-install "user/repo")                 ;; → /tmp/qlisp-pkg-<repo>/
(pkg-install-branch "user/repo" "dev")    ;; другая ветка
```

Внутри (`pkg-install-branch`):
1. `http-get` на `https://raw.githubusercontent.com/<repo>/<branch>/pkg.qlsp`.
2. Парсинг манифеста (список файлов).
3. `mkdir -p /tmp/qlisp-pkg-<repo>` + `spit` манифеста.
4. Для каждого файла в манифесте: `http-get` + `spit`.

Годится для простых библиотек-однофайловок; для всего остального — `qvalent`.

## 16.4 FFI: вызов C-библиотек

QLISP умеет загружать разделяемые библиотеки и вызывать C-функции напрямую. Реализация — `src/ffi/ffi.cpp`, примитивы регистрируются через `register_ffi_primitives`.

### Примитивы

| Примитив | Что делает |
|---|---|
| `(FFI-LOAD "libm.so.6")` | `dlopen`, возвращает индекс загруженной библиотеки |
| `(FFI-CALL "fn-name" "rtype" args...)` | вызвать C-функцию по имени |
| `(FFI-IMPORT "libfoo.so" "fn-name")` | загрузить и сделать символ видимым (low-level) |

### Сигнатуры FFI-CALL

Поддерживаемые комбинации `(rtype, arg-types)` (см. `prim_ffi_call` в `ffi.cpp`):

- `FLOAT <- FLOAT`
- `FLOAT <- FLOAT FLOAT`
- `FLOAT <- INT`
- `INT <- STRING`
- `TENSOR <- TENSOR TENSOR INT` — для kernel-вызовов (например, OpenCV)

Аргументы типов:

| Lisp-значение | atype | C-тип |
|---|---|---|
| `3.14` (flonum) | `FLOAT` | `float` |
| `42` (fixnum) | `INT` | `int` |
| `"hello"` (string) | `STRING` | `const char*` |
| Тензор | `TENSOR` | `float*` (или `uint8_t*` для U8) |

Примеры:

```lisp
(ffi-load "libm.so.6")              ;; → 0 (handle id)

;; синус: float sin(float)
(ffi-call "sin" "float" 0.0)        ;; → 0.0

;; atan2: float atan2(float, float)
(ffi-call "atan2" "float" 1.0 1.0)  ;; → 0.7853982...

;; strlen: int strlen(const char*)
(ffi-call "strlen" "int" "hello")   ;; → 5
```

> 🐍 **Python-аналогия.** `ctypes.CDLL("libm.so.6").sin(0.0)`. QLISP-вариант компактнее, но поддерживает ограниченный набор сигнатур (KISS-принцип, см. `AGENTS.md`). Для сложных случаев — `FFI-IMPORT` через `define-compiler-macro`-подход или ручной C-binding.

### Кросс-платформенность

На Linux/macOS — `dlopen`/`dlsym`/`dlclose` (`src/core/platform.hpp`). На Windows — `LoadLibraryA`/`GetProcAddress`/`FreeLibrary`. На macOS библиотеки имеют расширение `.dylib` (`SHARED_LIB_EXT`), на Windows — `.dll`.

### Известные ограничения

- Поддерживаются только перечисленные комбинации `(rtype, atypes)`; для других — `runtime_error: "ffi-call: unsupported signature"`.
- Нет автоматического маршалинга структур, массивов фиксированного размера, указателей на функции.
- Возвращаемое значение — `float` или `int` или `TENSOR` (для kernel-варианта). Строки-C как возврат пока не поддерживаются.