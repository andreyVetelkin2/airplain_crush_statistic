# Примеры использования API

## Примеры запросов к API

### 1. Проверка работоспособности

```bash
curl http://localhost:8000/api/health
```

**Ответ:**
```json
{
  "status": "ok",
  "message": "API работает нормально"
}
```

### 2. Получение списка переменных

```bash
curl http://localhost:8000/api/variables
```

**Ответ:**
```json
[
  {
    "code": "X1",
    "name": "Нарушения инструкций",
    "description": "Среднее количество нарушений инструкций пилотами",
    "type": "variable"
  },
  ...
]
```

### 3. Базовая симуляция (10 лет)

```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "years": 10,
    "dt": 0.1,
    "factor_multipliers": {
      "F1": 1.0,
      "F2": 1.0,
      "F3": 1.0,
      "F4": 1.0,
      "F5": 1.0
    }
  }'
```

### 4. Симуляция с увеличенным контролем государства

```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "years": 15,
    "dt": 0.1,
    "factor_multipliers": {
      "F1": 1.0,
      "F2": 1.0,
      "F3": 1.0,
      "F4": 1.0,
      "F5": 1.5
    }
  }'
```

### 5. Симуляция негативного сценария

```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "years": 10,
    "dt": 0.1,
    "factor_multipliers": {
      "F1": 0.8,
      "F2": 1.3,
      "F3": 1.0,
      "F4": 1.4,
      "F5": 0.7
    }
  }'
```

## Примеры на Python

### Базовый пример

```python
import requests
import json

# URL API
BASE_URL = "http://localhost:8000"

# Параметры симуляции
params = {
    "years": 10,
    "dt": 0.1,
    "factor_multipliers": {
        "F1": 1.0,
        "F2": 1.0,
        "F3": 1.0,
        "F4": 1.0,
        "F5": 1.0
    }
}

# Отправка запроса
response = requests.post(f"{BASE_URL}/api/simulate", json=params)

# Получение результатов
if response.status_code == 200:
    results = response.json()
    print(f"Время: {results['time'][:5]}")
    print(f"Катастрофы (X2): {results['X2'][:5]}")
else:
    print(f"Ошибка: {response.status_code}")
```

### Сравнение сценариев

```python
import requests
import matplotlib.pyplot as plt

BASE_URL = "http://localhost:8000"

# Сценарий 1: Базовый
scenario1 = {
    "years": 10,
    "dt": 0.1,
    "factor_multipliers": {"F1": 1.0, "F2": 1.0, "F3": 1.0, "F4": 1.0, "F5": 1.0}
}

# Сценарий 2: Усиленный контроль
scenario2 = {
    "years": 10,
    "dt": 0.1,
    "factor_multipliers": {"F1": 1.0, "F2": 1.0, "F3": 1.0, "F4": 1.0, "F5": 1.5}
}

# Выполнение симуляций
results1 = requests.post(f"{BASE_URL}/api/simulate", json=scenario1).json()
results2 = requests.post(f"{BASE_URL}/api/simulate", json=scenario2).json()

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(results1['time'], results1['X2'], label='Базовый сценарий', linewidth=2)
plt.plot(results2['time'], results2['X2'], label='Усиленный контроль (F5=1.5)', linewidth=2)
plt.xlabel('Годы от 2011')
plt.ylabel('Количество катастроф (нормированное)')
plt.title('Сравнение сценариев прогнозирования')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Параметрическое исследование

```python
import requests
import pandas as pd
import numpy as np

BASE_URL = "http://localhost:8000"

# Исследование влияния F5 (контроль государства)
f5_values = np.linspace(0.5, 2.0, 10)
final_disasters = []

for f5 in f5_values:
    params = {
        "years": 10,
        "dt": 0.1,
        "factor_multipliers": {"F1": 1.0, "F2": 1.0, "F3": 1.0, "F4": 1.0, "F5": f5}
    }
    
    results = requests.post(f"{BASE_URL}/api/simulate", json=params).json()
    # Берем значение катастроф на 10-м году
    final_disasters.append(results['X2'][-1])

# Создание DataFrame
df = pd.DataFrame({
    'F5 (Контроль государства)': f5_values,
    'Катастрофы на 10-м году': final_disasters
})

print(df)
```

## Примеры на JavaScript (Node.js)

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:8000';

// Базовая симуляция
async function runSimulation() {
    const params = {
        years: 10,
        dt: 0.1,
        factor_multipliers: {
            F1: 1.0,
            F2: 1.0,
            F3: 1.0,
            F4: 1.0,
            F5: 1.0
        }
    };
    
    try {
        const response = await axios.post(`${BASE_URL}/api/simulate`, params);
        const results = response.data;
        
        console.log('Результаты симуляции:');
        console.log('Время:', results.time.slice(0, 5));
        console.log('Катастрофы (X2):', results.X2.slice(0, 5));
    } catch (error) {
        console.error('Ошибка:', error.message);
    }
}

runSimulation();
```

## Формат ответа API /api/simulate

```json
{
  "time": [0.0, 0.1, 0.2, 0.3, ...],
  "X1": [1.0, 0.995, 0.991, ...],
  "X2": [1.0, 0.982, 0.965, ...],
  "X3": [1.0, 0.998, 0.996, ...],
  "X4": [1.0, 1.002, 1.004, ...],
  "X5": [1.0, 1.005, 1.009, ...],
  "X6": [1.0, 1.001, 1.003, ...],
  "X7": [1.0, 1.003, 1.005, ...],
  "X8": [1.0, 0.997, 0.994, ...]
}
```

Где:
- `time` - массив временных точек (годы от 2011)
- `X1-X8` - массивы значений переменных системы (нормированные)

## Обработка ошибок

### Неверные параметры

**Запрос:**
```json
{
  "years": 100,
  "dt": 0.1
}
```

**Ответ (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "years"],
      "msg": "ensure this value is less than or equal to 50",
      "type": "value_error.number.not_le"
    }
  ]
}
```

### Ошибка сервера

**Ответ (500):**
```json
{
  "detail": "Ошибка при выполнении симуляции: <описание ошибки>"
}
```

