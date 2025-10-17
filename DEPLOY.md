# Инструкция по бесплатному деплою

## Вариант 1: Render.com (Рекомендуется ⭐)

**Преимущества:**
- Очень простой процесс
- Бесплатный SSL сертификат
- Автоматический деплой из GitHub
- 750 часов/месяц бесплатно

### Шаги:

#### 1. Подготовка GitHub репозитория

```bash
# Инициализация Git (если еще не сделано)
git init

# Добавляем все файлы
git add .

# Коммитим
git commit -m "Initial commit: Aviation Safety Forecasting System"

# Создайте репозиторий на GitHub.com
# Затем подключите удаленный репозиторий
git remote add origin https://github.com/ваш-username/aviation-safety.git

# Отправьте код
git push -u origin main
```

#### 2. Деплой на Render

1. Зайдите на **https://render.com** и зарегистрируйтесь
2. Нажмите **"New +" → "Web Service"**
3. Подключите ваш GitHub репозиторий
4. Выберите репозиторий с проектом
5. Настройки:
   - **Name**: `aviation-safety` (или любое имя)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
6. Нажмите **"Create Web Service"**
7. Подождите 5-10 минут пока проект соберется
8. Получите ссылку типа: `https://aviation-safety.onrender.com`

**Готово!** ✅

---

## Вариант 2: Railway.app

**Преимущества:**
- 500 часов бесплатно в месяц
- Очень быстрый деплой
- Удобный интерфейс

### Шаги:

1. Зайдите на **https://railway.app**
2. Зарегистрируйтесь через GitHub
3. Нажмите **"New Project" → "Deploy from GitHub repo"**
4. Выберите ваш репозиторий
5. Railway автоматически определит Python проект
6. В настройках добавьте:
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
7. Нажмите **"Deploy"**
8. Получите ссылку на ваше приложение

---

## Вариант 3: Fly.io

**Преимущества:**
- Быстрые серверы
- Бесплатный tier с 3 VM
- Хорошая производительность

### Шаги:

1. Установите Fly CLI:
   ```bash
   # Windows (PowerShell)
   iwr https://fly.io/install.ps1 -useb | iex
   ```

2. Зарегистрируйтесь:
   ```bash
   fly auth signup
   ```

3. В корне проекта создайте файл `fly.toml`:
   ```toml
   app = "aviation-safety"
   
   [build]
   
   [env]
     PORT = "8000"
   
   [[services]]
     internal_port = 8000
     protocol = "tcp"
   
     [[services.ports]]
       handlers = ["http"]
       port = 80
   
     [[services.ports]]
       handlers = ["tls", "http"]
       port = 443
   ```

4. Создайте `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

5. Деплой:
   ```bash
   fly launch
   fly deploy
   ```

---

## Вариант 4: PythonAnywhere

**Преимущества:**
- Специально для Python
- Простая настройка
- Бесплатный tier

### Шаги:

1. Зарегистрируйтесь на **https://www.pythonanywhere.com**
2. Перейдите в **"Web" → "Add a new web app"**
3. Выберите **"Manual configuration"** и **Python 3.10**
4. В **"Code"** секции:
   - Клонируйте ваш GitHub репозиторий
   - Установите зависимости в virtualenv
5. Настройте WSGI файл для FastAPI
6. Перезапустите веб-приложение

---

## Вариант 5: Replit (Самый простой!)

**Преимущества:**
- Работает прямо в браузере
- Не нужен Git
- Мгновенный деплой

### Шаги:

1. Зайдите на **https://replit.com**
2. Создайте новый Repl → **Python**
3. Загрузите все файлы проекта (можно zip архивом)
4. В файле `.replit` укажите:
   ```
   run = "uvicorn backend.main:app --host 0.0.0.0 --port 8000"
   ```
5. Нажмите **"Run"**
6. Получите публичную ссылку

---

## Вариант 6: Vercel (для статики)

Vercel хорош для фронтенда, но для Python нужны дополнительные настройки.

### Альтернатива: Разделить фронт и бэк

1. **Фронтенд на Vercel/Netlify** (бесплатно)
2. **Бэкенд на Render/Railway** (бесплатно)

Создайте `vercel.json`:
```json
{
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://ваш-бэкенд.onrender.com/api/:path*"
    }
  ]
}
```

---

## Рекомендация по выбору

| Сервис | Сложность | Скорость | Бесплатный лимит | Рекомендация |
|--------|-----------|----------|------------------|--------------|
| **Render** | ⭐ Легко | Средняя | 750 ч/мес | ✅ **Лучший выбор** |
| **Railway** | ⭐ Легко | Быстрая | 500 ч/мес | ✅ Хорошо |
| **Replit** | ⭐⭐⭐ Очень легко | Средняя | Всегда онлайн | ✅ Для тестов |
| **Fly.io** | ⭐⭐ Средне | Очень быстрая | 3 VM | Для опытных |
| **PythonAnywhere** | ⭐⭐ Средне | Медленная | Ограничения | Устарело |

---

## Важные моменты

### 1. Переменные окружения

В настройках сервиса добавьте:
```
PORT=8000
PYTHON_VERSION=3.11
```

### 2. CORS для разделенного деплоя

Если фронт и бэк на разных доменах, обновите CORS в `backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ваш-фронтенд.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3. База данных (если понадобится)

Бесплатные варианты:
- PostgreSQL на Render (бесплатно)
- MongoDB Atlas (бесплатно)
- Supabase (бесплатно)

---

## Мониторинг после деплоя

1. **Логи**: Смотрите в панели управления сервиса
2. **Метрики**: Render/Railway показывают использование ресурсов
3. **Uptime**: Используйте https://uptimerobot.com (бесплатно)

---

## Ограничения бесплатных планов

⚠️ **Важно знать:**

1. **Render Free**: 
   - Засыпает после 15 минут неактивности
   - Первый запрос после сна может занять 30-60 секунд

2. **Railway Free**:
   - 500 часов/месяц = ~16 часов/день
   - Хватит для демо/тестирования

3. **Replit Free**:
   - Публичный код (все видят исходники)
   - Может засыпать

### Решение проблемы "засыпания":

Используйте **UptimeRobot** для пинга каждые 5 минут:
1. Зарегистрируйтесь на https://uptimerobot.com
2. Добавьте свой URL для мониторинга
3. Интервал проверки: 5 минут
4. Сервис будет постоянно "живым"

---

## Пример: Полный деплой на Render за 5 минут

```bash
# 1. Создайте репозиторий на GitHub
# Перейдите на github.com → New repository → создайте

# 2. Инициализация и пуш
cd C:\kushnikov1
git init
git add .
git commit -m "Aviation Safety System"
git branch -M main
git remote add origin https://github.com/username/aviation-safety.git
git push -u origin main

# 3. Зайдите на render.com
# 4. New → Web Service → Connect GitHub
# 5. Выберите репозиторий
# 6. Build Command: pip install -r requirements.txt
# 7. Start Command: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
# 8. Create Web Service
# 9. Ждите 5-10 минут
# 10. Готово! Получите ссылку типа: https://aviation-safety.onrender.com
```

**Всё! Ваше приложение доступно по всему миру! 🌍🚀**

---

## Полезные ссылки

- **Render**: https://render.com
- **Railway**: https://railway.app
- **Fly.io**: https://fly.io
- **Replit**: https://replit.com
- **PythonAnywhere**: https://www.pythonanywhere.com
- **UptimeRobot**: https://uptimerobot.com

## Поддержка

Если возникнут проблемы при деплое, проверьте:
1. Логи в панели управления сервиса
2. Правильность команд запуска
3. Версию Python (должна быть 3.8+)
4. Все ли зависимости в requirements.txt

