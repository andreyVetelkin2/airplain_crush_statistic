"""
Пример скрипта для запуска и тестирования модели системной динамики

Этот скрипт демонстрирует:
1. Базовую симуляцию
2. Использование solve_ivp и собственного RK4
3. Логирование шагов RK4
4. Проверку сходимости
5. Экспорт в CSV и построение графиков
6. Сценарии "what-if"
"""

import sys
from pathlib import Path

# Добавляем путь к backend
sys.path.insert(0, str(Path(__file__).parent))

from backend.model import AviationSafetyModel
import numpy as np


def main():
    """Основная функция для запуска симуляций"""
    
    print("=" * 80)
    print("МОДЕЛЬ СИСТЕМНОЙ ДИНАМИКИ: ПРОГНОЗИРОВАНИЕ БЕЗОПАСНОСТИ АТС")
    print("=" * 80)
    print()
    
    # ==================== БАЗОВЫЙ СЦЕНАРИЙ ====================
    print("1. БАЗОВЫЙ СЦЕНАРИЙ (RK4, 10 лет, шаг 0.1)")
    print("-" * 80)
    
    model = AviationSafetyModel(seed=42)
    
    results_basic = model.simulate(
        years=10,
        dt=0.1,
        method='rk4',
        log_rk4_steps=True,  # Логируем шаги RK4
        check_convergence=True,  # Проверяем сходимость
        export_csv=True,
        export_plots=True,
        output_dir='output'
    )
    
    print(f"✓ Симуляция завершена")
    print(f"  - Количество точек: {len(results_basic['time'])}")
    print(f"  - Финальные значения:")
    for i in range(1, 9):
        var_name = f'X{i}'
        final_val = results_basic[var_name][-1]
        print(f"    {var_name}: {final_val:.4f}")
    
    if 'convergence' in results_basic:
        conv = results_basic['convergence']
        print(f"  - {conv['message']}")
    
    if 'csv_path' in results_basic:
        print(f"  - CSV сохранен: {results_basic['csv_path']}")
    if 'plot_path' in results_basic:
        print(f"  - Графики сохранены: {results_basic['plot_path']}")
    
    print()
    
    # ==================== СЦЕНАРИЙ С SCIPY ====================
    print("2. СЦЕНАРИЙ С SCIPY (solve_ivp, метод RK45)")
    print("-" * 80)
    
    results_scipy = model.simulate(
        years=10,
        dt=0.1,
        method='scipy',
        export_csv=True,
        export_plots=True,
        output_dir='output'
    )
    
    print(f"✓ Симуляция с scipy завершена")
    print(f"  - Количество точек: {len(results_scipy['time'])}")
    print()
    
    # ==================== СЦЕНАРИЙ "WHAT-IF": УВЕЛИЧЕНИЕ F5 ====================
    print("3. СЦЕНАРИЙ 'WHAT-IF': УВЕЛИЧЕНИЕ F5 (контроль государства) в 1.5 раза")
    print("-" * 80)
    
    results_f5_increased = model.simulate(
        years=10,
        dt=0.1,
        factor_multipliers={'F5': 1.5},  # Увеличиваем F5
        method='rk4',
        export_csv=True,
        export_plots=True,
        output_dir='output'
    )
    
    print(f"✓ Симуляция с увеличенным F5 завершена")
    print(f"  - Сравнение финальных значений X2 (катастрофы):")
    print(f"    Базовый: {results_basic['X2'][-1]:.4f}")
    print(f"    F5×1.5:  {results_f5_increased['X2'][-1]:.4f}")
    print(f"    Изменение: {((results_f5_increased['X2'][-1] / results_basic['X2'][-1] - 1) * 100):.2f}%")
    print()
    
    # ==================== СЦЕНАРИЙ "WHAT-IF": ИЗМЕНЕНИЕ НАЧАЛЬНЫХ УСЛОВИЙ ====================
    print("4. СЦЕНАРИЙ 'WHAT-IF': ИЗМЕНЕНИЕ НАЧАЛЬНЫХ УСЛОВИЙ (X3 увеличен на 20%)")
    print("-" * 80)
    
    X0_modified = model.X0.copy()
    X0_modified[2] *= 1.2  # Увеличиваем X3 на 20%
    
    results_modified_init = model.simulate(
        years=10,
        dt=0.1,
        initial_conditions=X0_modified,
        method='rk4',
        export_csv=True,
        export_plots=True,
        output_dir='output'
    )
    
    print(f"✓ Симуляция с измененными начальными условиями завершена")
    print(f"  - Сравнение финальных значений X3 (коэффициент повторяемости):")
    print(f"    Базовый: {results_basic['X3'][-1]:.4f}")
    print(f"    X3(0)×1.2: {results_modified_init['X3'][-1]:.4f}")
    print()
    
    # ==================== ПРОВЕРКА СХОДИМОСТИ С РАЗНЫМИ ШАГАМИ ====================
    print("5. ПРОВЕРКА СХОДИМОСТИ (сравнение h=0.1 и h=0.05)")
    print("-" * 80)
    
    converged, error_norm, conv_results = model.check_convergence(
        t_span=(0, 10),
        h=0.1,
        tolerance=1e-3,
        method='rk4'
    )
    
    print(f"✓ Проверка сходимости завершена")
    print(f"  - Сходимость: {'ДА' if converged else 'НЕТ'}")
    print(f"  - L2-норма разности: {error_norm:.6e}")
    print(f"  - Порог: {conv_results['tolerance']:.6e}")
    print()
    
    # ==================== ЛОГИРОВАНИЕ ШАГОВ RK4 ====================
    print("6. ПРИМЕР ЛОГИРОВАНИЯ ШАГОВ RK4 (первые 3 шага)")
    print("-" * 80)
    
    if 'rk4_log' in results_basic:
        rk4_log = results_basic['rk4_log']
        n_steps_to_show = min(3, len(rk4_log['t']))
        
        for i in range(n_steps_to_show):
            t = rk4_log['t'][i]
            k1 = rk4_log['k1'][i]
            k2 = rk4_log['k2'][i]
            k3 = rk4_log['k3'][i]
            k4 = rk4_log['k4'][i]
            
            print(f"  Шаг {i+1} (t = {t:.2f}):")
            print(f"    k1 = [{', '.join(f'{x:.4f}' for x in k1[:3])}...]")
            print(f"    k2 = [{', '.join(f'{x:.4f}' for x in k2[:3])}...]")
            print(f"    k3 = [{', '.join(f'{x:.4f}' for x in k3[:3])}...]")
            print(f"    k4 = [{', '.join(f'{x:.4f}' for x in k4[:3])}...]")
            print()
    
    # ==================== ИТОГОВАЯ СТАТИСТИКА ====================
    print("=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)
    
    print("\nБазовый сценарий - финальные значения (t=10 лет):")
    print(f"  X1 (нарушения инструкций):     {results_basic['X1'][-1]:.4f}")
    print(f"  X2 (количество катастроф):     {results_basic['X2'][-1]:.4f}")
    print(f"  X3 (коэффициент повторяемости): {results_basic['X3'][-1]:.4f}")
    print(f"  X4 (доля частных судов):       {results_basic['X4'][-1]:.4f}")
    print(f"  X5 (метеослужбы):              {results_basic['X5'][-1]:.4f}")
    print(f"  X6 (контроль контрафакта):     {results_basic['X6'][-1]:.4f}")
    print(f"  X7 (возраст судов):            {results_basic['X7'][-1]:.4f}")
    print(f"  X8 (квалификация персонала):   {results_basic['X8'][-1]:.4f}")
    
    print("\n✓ Все симуляции завершены успешно!")
    print(f"✓ Результаты сохранены в директории: output/")
    print()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

