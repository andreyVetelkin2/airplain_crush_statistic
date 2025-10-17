#!/bin/bash

echo "========================================"
echo "Aviation Safety Forecasting System"
echo "========================================"
echo ""

# Проверка установки Python
if ! command -v python3 &> /dev/null
then
    echo "Ошибка: Python 3 не установлен"
    exit 1
fi

echo "Установка зависимостей..."
pip install -r requirements.txt

echo ""
echo "Запуск сервера..."
echo ""
echo "Откройте браузер и перейдите по адресу: http://localhost:8000"
echo ""

cd backend
python3 main.py

