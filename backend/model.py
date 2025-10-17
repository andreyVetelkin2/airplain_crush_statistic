"""
Математическая модель системной динамики для прогнозирования 
характеристик безопасности авиационных транспортных систем
"""

import numpy as np
from typing import List, Tuple, Dict


class AviationSafetyModel:
    """
    Модель прогнозирования безопасности АТС на основе системной динамики
    
    Переменные системы:
    X1(t) - среднее количество нарушений инструкций пилотами
    X2(t) - количество катастроф
    X3(t) - коэффициент повторяемости авиационных происшествий
    X4(t) - доля частных судов в авиации
    X5(t) - количество сотрудников метеорологических служб
    X6(t) - показатель активности органов контроля за контрафактом
    X7(t) - средний возраст воздушных судов
    X8(t) - средняя квалификация персонала
    
    Внешние факторы:
    F1(t) - средний лётный стаж пилотов
    F2(t) - доля иностранных воздушных судов
    F3(t) - средняя выработка ресурса до списания
    F4(t) - стоимость авиационного топлива
    F5(t) - количество нормативно-правовых актов (контроль государства)
    """
    
    def __init__(self):
        # Начальные условия (нормированные значения для 2011 года)
        self.X0 = np.array([
            1.0,  # X1 - нарушения инструкций
            1.0,  # X2 - количество катастроф
            1.0,  # X3 - коэффициент повторяемости
            1.0,  # X4 - доля частных судов
            1.0,  # X5 - метеослужбы
            1.0,  # X6 - контроль контрафакта
            1.0,  # X7 - возраст судов
            1.0   # X8 - квалификация персонала
        ])
    
    def external_factors(self, t: float, factor_multipliers: Dict[str, float]) -> Tuple[float, float, float, float, float]:
        """
        Вычисление внешних факторов в зависимости от времени
        
        Args:
            t: время (годы от 2011)
            factor_multipliers: множители для изменения внешних факторов
        
        Returns:
            F1, F2, F3, F4, F5
        """
        # Базовые значения внешних факторов (из статьи)
        F1 = (1.0 - 0.02 * t) * factor_multipliers.get('F1', 1.0)  # лётный стаж
        F2 = (1.0 + 0.03 * t) * factor_multipliers.get('F2', 1.0)  # доля иностранных судов
        F3 = (1.0 + 0.01 * t) * factor_multipliers.get('F3', 1.0)  # выработка ресурса
        F4 = (1.0 + 0.05 * t) * factor_multipliers.get('F4', 1.0)  # стоимость топлива
        F5 = (1.0 + 0.04 * t) * factor_multipliers.get('F5', 1.0)  # НПА (контроль государства)
        
        return F1, F2, F3, F4, F5
    
    def system_equations(self, X: np.ndarray, t: float, factor_multipliers: Dict[str, float]) -> np.ndarray:
        """
        Система дифференциальных уравнений (на основе уравнений из статьи)
        
        Args:
            X: вектор переменных [X1, X2, X3, X4, X5, X6, X7, X8]
            t: время
            factor_multipliers: множители для внешних факторов
        
        Returns:
            dX/dt - производные переменных
        """
        X1, X2, X3, X4, X5, X6, X7, X8 = X
        F1, F2, F3, F4, F5 = self.external_factors(t, factor_multipliers)
        
        # Система дифференциальных уравнений (упрощённая версия на основе статьи)
        dX = np.zeros(8)
        
        # Упрощенная линейная модель с балансировкой
        # Константы для стабилизации
        baseline = 0.02  # Базовая динамика
        
        # dX1/dt - нарушения инструкций
        dX1 = baseline * (F1 - X1 * 0.1 - X2 * 0.05)
        
        # dX2/dt - количество катастроф (основное уравнение)
        dX2 = baseline * (
            0.3 * X3 + 0.2 * X4 + 0.15 * X6 + 0.1 * X7  # Факторы риска
            + 0.2 * (F2 - 1.0) + 0.15 * (F4 - 1.0)  # Внешние негативные факторы
            - 0.4 * (F5 - 1.0) - 0.3 * (F1 - 1.0)  # Внешние позитивные факторы
            - 0.1 * X5 - 0.1 * X8  # Внутренние позитивные факторы
        )
        
        # dX3/dt - коэффициент повторяемости
        dX3 = baseline * (-0.15 * X1 - 0.1 * (F3 - 1.0) + 0.1 * (F4 - 1.0) - 0.1 * (F5 - 1.0))
        
        # dX4/dt - доля частных судов
        dX4 = baseline * (0.05 * X2 - 0.1 * (F5 - 1.0) + 0.05 * (F2 - 1.0))
        
        # dX5/dt - метеорологические службы
        dX5 = baseline * (0.1 * X2 + 0.2 * (F5 - 1.0))
        
        # dX6/dt - контроль контрафакта
        dX6 = baseline * (0.05 * X2 - 0.15 * (F5 - 1.0) + 0.05 * (F2 - 1.0))
        
        # dX7/dt - возраст судов
        dX7 = baseline * (0.03 * X2 + 0.03 * X3 - 0.1 * (F5 - 1.0))
        
        # dX8/dt - квалификация персонала
        dX8 = baseline * (0.15 * (F1 - 1.0) - 0.05 * X2 + 0.1 * (F5 - 1.0))
        
        dX[0] = dX1
        dX[1] = dX2
        dX[2] = dX3
        dX[3] = dX4
        dX[4] = dX5
        dX[5] = dX6
        dX[6] = dX7
        dX[7] = dX8
        
        return dX
    
    def runge_kutta_4(self, t_start: float, t_end: float, dt: float, 
                      factor_multipliers: Dict[str, float], 
                      initial_conditions: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Решение системы дифференциальных уравнений методом Рунге-Кутта 4-го порядка
        
        Args:
            t_start: начальное время
            t_end: конечное время
            dt: шаг интегрирования
            factor_multipliers: множители для внешних факторов
            initial_conditions: начальные условия (если None, используются стандартные)
        
        Returns:
            t_values: массив значений времени
            X_values: массив значений переменных системы
        """
        # Инициализация
        t_values = np.arange(t_start, t_end + dt, dt)
        n_steps = len(t_values)
        X_values = np.zeros((n_steps, 8))
        
        # Начальные условия
        if initial_conditions is not None:
            X_values[0] = initial_conditions
        else:
            X_values[0] = self.X0
        
        # Метод Рунге-Кутта 4-го порядка
        for i in range(n_steps - 1):
            X_current = X_values[i]
            t_current = t_values[i]
            
            # Коэффициенты Рунге-Кутта
            k1 = dt * self.system_equations(X_current, t_current, factor_multipliers)
            k2 = dt * self.system_equations(X_current + 0.5 * k1, t_current + 0.5 * dt, factor_multipliers)
            k3 = dt * self.system_equations(X_current + 0.5 * k2, t_current + 0.5 * dt, factor_multipliers)
            k4 = dt * self.system_equations(X_current + k3, t_current + dt, factor_multipliers)
            
            # Новое значение
            X_values[i + 1] = X_current + (k1 + 2*k2 + 2*k3 + k4) / 6
            
            # Ограничение отрицательных значений (физический смысл)
            X_values[i + 1] = np.maximum(X_values[i + 1], 0.01)
        
        return t_values, X_values
    
    def simulate(self, years: int = 10, dt: float = 0.1, 
                 factor_multipliers: Dict[str, float] = None,
                 initial_conditions: np.ndarray = None) -> Dict:
        """
        Запуск симуляции
        
        Args:
            years: количество лет для прогноза
            dt: шаг интегрирования
            factor_multipliers: множители для внешних факторов
            initial_conditions: начальные условия
        
        Returns:
            Словарь с результатами симуляции
        """
        if factor_multipliers is None:
            factor_multipliers = {}
        
        t_values, X_values = self.runge_kutta_4(0, years, dt, factor_multipliers, initial_conditions)
        
        return {
            'time': t_values.tolist(),
            'X1': X_values[:, 0].tolist(),  # нарушения инструкций
            'X2': X_values[:, 1].tolist(),  # количество катастроф
            'X3': X_values[:, 2].tolist(),  # коэффициент повторяемости
            'X4': X_values[:, 3].tolist(),  # доля частных судов
            'X5': X_values[:, 4].tolist(),  # метеослужбы
            'X6': X_values[:, 5].tolist(),  # контроль контрафакта
            'X7': X_values[:, 6].tolist(),  # возраст судов
            'X8': X_values[:, 7].tolist(),  # квалификация персонала
        }

