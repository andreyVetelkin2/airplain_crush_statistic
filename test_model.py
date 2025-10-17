"""
Тестовый скрипт для проверки математической модели
"""

import sys
sys.path.append('backend')

from model import AviationSafetyModel
import json

print("="*50)
print("Тест математической модели")
print("="*50)

model = AviationSafetyModel()

# Тест 1: Базовая симуляция
print("\nТест 1: Базовая симуляция (все факторы = 1.0)")
print("-"*50)
results = model.simulate(years=10, dt=0.1, factor_multipliers={})
print(f"Количество точек: {len(results['time'])}")
print(f"Время начало: {results['time'][0]}, конец: {results['time'][-1]}")
print(f"X2 начало: {results['X2'][0]:.4f}, конец: {results['X2'][-1]:.4f}")
print(f"Изменение X2: {(results['X2'][-1] - results['X2'][0]):.4f}")

# Тест 2: Увеличенный контроль государства
print("\nТест 2: Увеличенный контроль (F5 = 1.5)")
print("-"*50)
results2 = model.simulate(years=10, dt=0.1, factor_multipliers={'F5': 1.5})
print(f"X2 начало: {results2['X2'][0]:.4f}, конец: {results2['X2'][-1]:.4f}")
print(f"Изменение X2: {(results2['X2'][-1] - results2['X2'][0]):.4f}")

# Тест 3: Ослабленный контроль
print("\nТест 3: Ослабленный контроль (F5 = 0.5)")
print("-"*50)
results3 = model.simulate(years=10, dt=0.1, factor_multipliers={'F5': 0.5})
print(f"X2 начало: {results3['X2'][0]:.4f}, конец: {results3['X2'][-1]:.4f}")
print(f"Изменение X2: {(results3['X2'][-1] - results3['X2'][0]):.4f}")

# Сравнение
print("\n" + "="*50)
print("Сравнение конечных значений X2:")
print("="*50)
print(f"Базовый (F5=1.0):    {results['X2'][-1]:.4f}")
print(f"Усиленный (F5=1.5):  {results2['X2'][-1]:.4f}")
print(f"Ослабленный (F5=0.5): {results3['X2'][-1]:.4f}")

# Вывод нескольких точек для проверки
print("\n" + "="*50)
print("Первые 10 значений X2 (базовый сценарий):")
print("="*50)
for i in range(min(10, len(results['X2']))):
    print(f"t={results['time'][i]:.1f}: X2={results['X2'][i]:.4f}")

print("\nТест завершен!")

