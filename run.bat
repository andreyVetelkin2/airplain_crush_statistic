@echo off
echo ========================================
echo Aviation Safety Forecasting System
echo ========================================
echo.

REM Проверка установки Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Ошибка: Python не установлен или не добавлен в PATH
    pause
    exit /b 1
)

REM Создание venv если его нет
if not exist venv (
    echo Создание виртуального окружения...
    python -m venv venv
)

REM Активация venv
echo Активация виртуального окружения...
call venv\Scripts\activate

REM Установка зависимостей
echo Установка зависимостей...
pip install setuptools wheel
pip install -r requirements.txt

echo.
echo Запуск сервера...
echo.
echo Откройте браузер и перейдите по адресу: http://localhost:8000
echo Для остановки сервера нажмите Ctrl+C
echo.

cd backend
python main.py

