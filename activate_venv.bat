@echo off
echo Активация виртуального окружения...
call venv\Scripts\activate.bat
echo.
echo Виртуальное окружение активировано!
echo Теперь вы можете запускать команды Python в изолированной среде.
echo.
echo Для запуска сервера выполните:
echo   cd backend
echo   python main.py
echo.
echo Для деактивации выполните: deactivate
echo.
cmd /k

