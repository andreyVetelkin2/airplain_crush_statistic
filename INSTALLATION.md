# Инструкция по установке

## Полная пошаговая инструкция для Windows

### Шаг 1: Установка Python

1. Скачайте Python 3.8 или выше с официального сайта: https://www.python.org/downloads/
2. Запустите установщик
3. **ВАЖНО:** Поставьте галочку "Add Python to PATH"
4. Нажмите "Install Now"
5. Проверьте установку, открыв командную строку и выполнив:
   ```
   python --version
   ```

### Шаг 2: Создание виртуального окружения

Откройте PowerShell или командную строку в папке проекта `C:\kushnikov1`:

```powershell
# Создание виртуального окружения
python -m venv venv

# Активация виртуального окружения (PowerShell)
.\venv\Scripts\activate

# Или для cmd
venv\Scripts\activate.bat
```

После активации в начале строки появится `(venv)`.

### Шаг 3: Установка зависимостей

```powershell
# Обновление pip
python -m pip install --upgrade pip

# Установка базовых инструментов
pip install setuptools wheel

# Установка зависимостей проекта
pip install -r requirements.txt
```

### Шаг 4: Запуск сервера

```powershell
cd backend
python main.py
```

Или просто запустите файл **`run.bat`** двойным кликом!

### Шаг 5: Открытие в браузере

Откройте браузер и перейдите по адресу:
```
http://localhost:8000
```

## Быстрый запуск (после первой установки)

### Вариант 1: Через файл .bat
Просто запустите `run.bat` двойным кликом.

### Вариант 2: Вручную

1. Откройте PowerShell в папке проекта
2. Активируйте виртуальное окружение:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
3. Запустите сервер:
   ```powershell
   cd backend
   python main.py
   ```

### Вариант 3: Через uvicorn

```powershell
.\venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## Работа с виртуальным окружением

### Активация

**PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**CMD:**
```cmd
venv\Scripts\activate.bat
```

### Деактивация

```
deactivate
```

### Проверка установленных пакетов

```powershell
pip list
```

### Обновление зависимостей

```powershell
pip install --upgrade -r requirements.txt
```

## Решение возможных проблем

### Ошибка: "cannot be loaded because running scripts is disabled"

Это проблема политики выполнения PowerShell. Решение:

1. Откройте PowerShell от имени администратора
2. Выполните:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
3. Подтвердите изменение (введите Y)

### Ошибка: "pip is not recognized"

Используйте:
```powershell
python -m pip install -r requirements.txt
```

### Ошибка: "Address already in use" (порт 8000 занят)

Запустите на другом порту:
```powershell
uvicorn backend.main:app --port 8001
```

### Ошибка при установке numpy/scipy

Установите Microsoft Visual C++ Build Tools:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

Или используйте предкомпилированные пакеты:
```powershell
pip install --only-binary :all: numpy scipy
```

### Браузер не открывается автоматически

Откройте вручную и введите адрес: http://localhost:8000

## Структура виртуального окружения

```
kushnikov1/
├── venv/                    # Виртуальное окружение (НЕ коммитить!)
│   ├── Scripts/             # Исполняемые файлы
│   │   ├── activate.bat     # Активация (CMD)
│   │   ├── Activate.ps1     # Активация (PowerShell)
│   │   ├── python.exe       # Python в venv
│   │   └── pip.exe          # pip в venv
│   ├── Lib/                 # Библиотеки Python
│   └── Include/             # Заголовочные файлы
├── backend/                 # Код бэкенда
├── frontend/                # Код фронтенда
├── requirements.txt         # Зависимости проекта
└── run.bat                  # Скрипт быстрого запуска
```

## Проверка работоспособности

### 1. Проверка API

```powershell
curl http://localhost:8000/api/health
```

Ожидаемый ответ:
```json
{"status":"ok","message":"API работает нормально"}
```

### 2. Проверка Swagger документации

Откройте в браузере:
```
http://localhost:8000/docs
```

### 3. Тестовая симуляция

```powershell
curl -X POST http://localhost:8000/api/simulate `
  -H "Content-Type: application/json" `
  -d '{"years":5,"dt":0.1}'
```

## Переменные окружения (опционально)

Создайте файл `.env` в корне проекта для дополнительных настроек:

```
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

## Обновление проекта

```powershell
# Активировать venv
.\venv\Scripts\Activate.ps1

# Обновить зависимости
pip install --upgrade -r requirements.txt

# Перезапустить сервер
cd backend
python main.py
```

## Удаление виртуального окружения

Если нужно пересоздать venv:

```powershell
# Деактивировать
deactivate

# Удалить папку venv
Remove-Item -Recurse -Force venv

# Создать заново
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Для разработчиков

### Установка дополнительных инструментов

```powershell
pip install pytest black flake8 mypy
```

### Запуск в режиме разработки с автоперезагрузкой

```powershell
uvicorn backend.main:app --reload
```

### Заморозка зависимостей

Если добавили новые пакеты:
```powershell
pip freeze > requirements.txt
```

## Полезные команды

```powershell
# Показать путь к Python в venv
where python

# Показать установленные пакеты
pip list

# Показать устаревшие пакеты
pip list --outdated

# Обновить конкретный пакет
pip install --upgrade fastapi

# Удалить пакет
pip uninstall package_name

# Очистка кеша pip
pip cache purge
```

## Следующие шаги

После успешной установки:

1. Изучите интерфейс по адресу http://localhost:8000
2. Прочитайте [QUICKSTART.md](QUICKSTART.md) для первых экспериментов
3. Изучите [API_EXAMPLES.md](API_EXAMPLES.md) для программного доступа
4. Прочитайте [MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md) для понимания модели

---

**Готово! Система установлена и готова к работе! 🎉**

