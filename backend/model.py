import numpy as np
from typing import Dict, Optional, Tuple, List


class AviationSafetyModel:
    """
    Стабильная 8-переменная модель системной динамики с 5 внешними факторами.

    Переменные:
      X1 — нарушения пилотов
      X2 — количество катастроф
      X3 — повторяемость происшествий
      X4 — доля частных судов
      X5 — сотрудники метеослужб
      X6 — контроль контрафакта
      X7 — возраст судов
      X8 — квалификация персонала

    Внешние факторы:
      F1 — контроль государства
      F2 — инфраструктура/топливо/снабжение
      F3 — условия среды
      F4 — доля иностранных самолётов
      F5 — законодательство/регуляторика
    """

    def __init__(self, params: Optional[Dict[str, float]] = None, equation_coefs: Optional[List[List[float]]] = None):
        """
        Инициализация модели.
        
        Args:
            params: словарь параметров модели (для обратной совместимости)
            equation_coefs: список коэффициентов для уравнений f1-f20. 
                          Каждый элемент - список [a, b, c, d] для полинома a*x³ + b*x² + c*x + d
                          Если None, используются значения по умолчанию или старая модель.
        """
        defaults = {
            "a1": 0.15,
            "b1": 0.08,
            "c1": 0.05,
            "a2": 0.25,
            "b2": 0.18,
            "c2": 0.12,
            "d2": 0.10,
            "e2": 0.08,
            "f2": 0.07,
            "g2": 0.10,
            "a3": 0.20,
            "b3": 0.12,
            "c3": 0.08,
            "a4": 0.14,
            "b4": 0.10,
            "c4": 0.12,
            "a5": 0.20,
            "b5": 0.10,
            "a6": 0.18,
            "b6": 0.09,
            "a7": 0.10,
            "b7": 0.05,
            "a8": 0.22,
            "b8": 0.11,
        }
        self.params = {**defaults, **(params or {})}
        self.X0 = np.array([0.8, 0.9, 0.6, 0.4, 0.7, 0.6, 0.5, 0.8], dtype=float)
        self.factor_multipliers: Dict[str, float] = {}
        self.factor_bounds: Dict[str, Tuple[float, float]] = {
            "F1": (0.3, 2.0),
            "F2": (0.3, 2.0),
            "F3": (0.3, 2.0),
            "F4": (0.3, 2.0),
            "F5": (0.3, 2.0),
        }
        
        # Коэффициенты для уравнений f1-f20
        # Структура: каждая функция fi имеет коэффициенты [a, b, c, d] для полинома a*x³ + b*x² + c*x + d
        # Для смешанных уравнений структура зависит от переменных
        self.equation_coefs = equation_coefs  # Список из 20 списков по 4 коэффициента
        self.use_equations = equation_coefs is not None and len(equation_coefs) >= 18

    # ---------------------- Внешние факторы ---------------------- #

    def _clip_factor(self, name: str, value: float) -> float:
        low, high = self.factor_bounds[name]
        return float(np.clip(value, low, high))

    def _factors(
        self,
        t: float,
        factor_multipliers: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float, float, float]:
        multipliers = factor_multipliers or self.factor_multipliers or {}

        base_F1 = 0.6 + 0.02 * t
        base_F2 = 0.9 + 0.01 * t
        base_F3 = 0.7 + 0.015 * t
        base_F4 = 0.5 + 0.01 * t
        base_F5 = 0.65 + 0.025 * t

        F1 = self._clip_factor("F1", base_F1 * multipliers.get("F1", 1.0))
        F2 = self._clip_factor("F2", base_F2 * multipliers.get("F2", 1.0))
        F3 = self._clip_factor("F3", base_F3 * multipliers.get("F3", 1.0))
        F4 = self._clip_factor("F4", base_F4 * multipliers.get("F4", 1.0))
        F5 = self._clip_factor("F5", base_F5 * multipliers.get("F5", 1.0))

        return F1, F2, F3, F4, F5

    # -------------------- Вычисление полиномов f1-f20 ------------------- #
    
    def _compute_polynomial(self, coefs: List[float], x: float) -> float:
        """
        Вычисляет значение полинома a*x³ + b*x² + c*x + d
        
        Args:
            coefs: [a, b, c, d] - коэффициенты полинома
            x: значение переменной
            
        Returns:
            значение полинома
        """
        if len(coefs) < 4:
            return 0.0
        a, b, c, d = coefs[0], coefs[1], coefs[2], coefs[3]
        return a * x**3 + b * x**2 + c * x + d
    
    def _compute_polynomial_from_var(self, func_num: int, var_value: float, X: np.ndarray) -> float:
        """
        Вычисляет полином fi от заданного значения переменной.
        Использует коэффициенты функции fi для вычисления полинома от var_value.
        
        Args:
            func_num: номер функции (1-8)
            var_value: значение переменной, от которой вычисляем полином
            X: вектор переменных (не используется, но нужен для совместимости)
            
        Returns:
            значение полинома
        """
        if not self.use_equations or func_num < 1 or func_num > len(self.equation_coefs):
            return 0.0
        coefs = self.equation_coefs[func_num - 1]
        return self._compute_polynomial(coefs, var_value)
    
    def _compute_f(self, i: int, X: np.ndarray) -> float:
        """
        Вычисляет значение функции fi согласно структуре из интерфейса.
        
        f1-f8: полиномы от одной переменной (X1-X8)
        f9-f20: смешанные полиномы
        
        Args:
            i: номер функции (1-20, индексация с 1)
            X: вектор переменных [X1, X2, ..., X8]
            
        Returns:
            значение функции fi
        """
        if not self.use_equations or i < 1 or i > len(self.equation_coefs):
            return 0.0
            
        coefs = self.equation_coefs[i - 1]  # Индексация с 0
        
        # f1-f8: полиномы от одной переменной
        if 1 <= i <= 8:
            x = X[i - 1]  # f1 от X1, f2 от X2, и т.д.
            return self._compute_polynomial(coefs, x)
        
        # f9-f20: смешанные полиномы
        # Структура из HTML:
        # f9 = a9*X1³ + b9*X2² + c9*X3 + d9
        # f10 = a10*X4³ + b10*X5² + c10*X6 + d10
        # f11 = a11*X7³ + b11*X8² + c11*X1 + d11
        # f12 = a12*X2³ + b12*X3² + c12*X4 + d12
        # f13 = a13*X5³ + b13*X6² + c13*X7 + d13
        # f14 = a14*X8³ + b14*X1² + c14*X2 + d14
        # f15 = a15*X3³ + b15*X4² + c15*X5 + d15
        # f16 = a16*X6³ + b16*X7² + c16*X8 + d16
        # f17 = a17*X1³ + b17*X3² + c17*X5 + d17
        # f18 = a18*X2³ + b18*X4² + c18*X6 + d18
        # f19 = a19*X7³ + b19*X1² + c19*X8 + d19
        # f20 = a20*X4³ + b20*X2² + c20*X3 + d20
        
        if i == 9:
            return coefs[0] * X[0]**3 + coefs[1] * X[1]**2 + coefs[2] * X[2] + coefs[3]
        elif i == 10:
            return coefs[0] * X[3]**3 + coefs[1] * X[4]**2 + coefs[2] * X[5] + coefs[3]
        elif i == 11:
            return coefs[0] * X[6]**3 + coefs[1] * X[7]**2 + coefs[2] * X[0] + coefs[3]
        elif i == 12:
            return coefs[0] * X[1]**3 + coefs[1] * X[2]**2 + coefs[2] * X[3] + coefs[3]
        elif i == 13:
            return coefs[0] * X[4]**3 + coefs[1] * X[5]**2 + coefs[2] * X[6] + coefs[3]
        elif i == 14:
            return coefs[0] * X[7]**3 + coefs[1] * X[0]**2 + coefs[2] * X[1] + coefs[3]
        elif i == 15:
            return coefs[0] * X[2]**3 + coefs[1] * X[3]**2 + coefs[2] * X[4] + coefs[3]
        elif i == 16:
            return coefs[0] * X[5]**3 + coefs[1] * X[6]**2 + coefs[2] * X[7] + coefs[3]
        elif i == 17:
            return coefs[0] * X[0]**3 + coefs[1] * X[2]**2 + coefs[2] * X[4] + coefs[3]
        elif i == 18:
            return coefs[0] * X[1]**3 + coefs[1] * X[3]**2 + coefs[2] * X[5] + coefs[3]
        elif i == 19:
            return coefs[0] * X[6]**3 + coefs[1] * X[0]**2 + coefs[2] * X[7] + coefs[3]
        elif i == 20:
            return coefs[0] * X[3]**3 + coefs[1] * X[1]**2 + coefs[2] * X[2] + coefs[3]
        
        return 0.0

    # -------------------- Правая часть системы ------------------- #

    def _rhs(self, t: float, X: np.ndarray) -> np.ndarray:
        X1, X2, X3, X4, X5, X6, X7, X8 = X
        F1, F2, F3, F4, F5 = self._factors(t)
        
        # Если используются уравнения f1-f20 из интерфейса
        if self.use_equations:
            # Система дифференциальных уравнений согласно tz.txt:
            # dX1/dt = F3(t) - f1(X2(t))f2(X3(t))f3(X4(t))
            # dX2/dt = (F1(t)+F2(t)+F4(t))f4(X4(t))f5(X6(t))f6(X7(t)) - (F3(t)+F5(t))f7(X8(t))
            # dX3/dt = F4(t)f8(X7(t)) - (F3(t)+F5(t))f9(X1(t))
            # dX4/dt = F2(t)f10(X2(t))f11(X7(t)) - (F3(t)+F5(t))f12(X1(t))
            # dX5/dt = f13(X2(t)) - F5(t)
            # dX6/dt = F2(t)f14(X2(t)) - F5(t)
            # dX7/dt = F2(t)f15(X2(t))f16(X3(t))f17(X4(t)) - (F3(t)+F5(t))
            # dX8/dt = F5(t) - f18(X2(t))
            
            # Вычисляем функции f1-f8 от нужных переменных согласно уравнениям
            # f1 используется с X2, f2 с X3, f3 с X4, и т.д.
            # Но в интерфейсе f1 определен как полином от X1, поэтому используем общий метод вычисления
            
            # Для f1-f8: берем коэффициенты соответствующей функции и вычисляем от нужной переменной
            # f1(X2), f2(X3), f3(X4), f4(X4), f5(X6), f6(X7), f7(X8), f8(X7)
            f1_X2 = self._compute_polynomial_from_var(1, X2, X)
            f2_X3 = self._compute_polynomial_from_var(2, X3, X)
            f3_X4 = self._compute_polynomial_from_var(3, X4, X)
            f4_X4 = self._compute_polynomial_from_var(4, X4, X)
            f5_X6 = self._compute_polynomial_from_var(5, X6, X)
            f6_X7 = self._compute_polynomial_from_var(6, X7, X)
            f7_X8 = self._compute_polynomial_from_var(7, X8, X)
            f8_X7 = self._compute_polynomial_from_var(8, X7, X)
            
            # Для f10-f12: согласно tz.txt f10 от X2, f11 от X7, f12 от X1
            # Но в интерфейсе это смешанные полиномы, используем их как есть
            # f10 = a10*X4³ + b10*X5² + c10*X6 + d10 (из HTML, но нужна от X2)
            # Для совместимости: если f10 определена как смешанная, используем её,
            # но для правильного соответствия tz.txt нужно f10(X2), f11(X7), f12(X1)
            
            # Используем смешанные функции f10-f12 как определено в интерфейсе
            # Если нужна точная соответствие tz.txt, можно использовать f10 для X2 и т.д.
            f10_value = self._compute_f(10, X)  # Смешанный полином от X4, X5, X6
            f11_value = self._compute_f(11, X)  # Смешанный полином от X7, X8, X1
            f12_value = self._compute_f(12, X)  # Смешанный полином от X2, X3, X4
            
            # Для f13-f14 нужны от X2: используем смешанные функции или полиномы
            f13_value = self._compute_f(13, X)  # Смешанный полином от X5, X6, X7
            f14_value = self._compute_f(14, X)  # Смешанный полином от X8, X1, X2
            
            # Для dX5: f13(X2) согласно tz.txt, но f13 в интерфейсе - смешанный
            # Используем полином от X2 с коэффициентами f13, но берём значение от X2
            # Альтернатива: использовать смешанный полином как есть
            f13_X2 = self._compute_polynomial_from_var(13, X2, X) if len(self.equation_coefs) > 12 else f13_value
            f14_X2 = self._compute_polynomial_from_var(14, X2, X) if len(self.equation_coefs) > 13 else f14_value
            
            dX1 = F3 - f1_X2 * f2_X3 * f3_X4
            dX2 = (F1 + F2 + F4) * f4_X4 * f5_X6 * f6_X7 - (F3 + F5) * f7_X8
            dX3 = F4 * f8_X7 - (F3 + F5) * self._compute_f(9, X)
            dX4 = F2 * f10_value * f11_value - (F3 + F5) * f12_value
            dX5 = f13_X2 - F5
            dX6 = F2 * f14_X2 - F5
            dX7 = F2 * self._compute_f(15, X) * self._compute_f(16, X) * self._compute_f(17, X) - (F3 + F5)
            dX8 = F5 - self._compute_polynomial_from_var(18, X2, X) if len(self.equation_coefs) > 17 else (F5 - X2 * 0.02)  # f18(X2) согласно tz.txt
        else:
            # Старая модель (для обратной совместимости)
            p = self.params
            dX1 = p["a1"] * F3 - p["b1"] * F5 - p["c1"] * X8
            dX2 = (
                p["a2"] * X1
                + p["b2"] * X3
                + p["c2"] * X4
                - p["d2"] * X5
                - p["e2"] * X6
                - p["f2"] * X8
                - p["g2"] * F5
            )
            dX3 = p["a3"] * X2 + p["b3"] * X1 - p["c3"] * X8
            dX4 = p["a4"] * X2 + p["b4"] * X3 - p["c4"] * F5
            dX5 = p["a5"] * F2 - p["b5"] * X5
            dX6 = p["a6"] * F2 - p["b6"] * X6
            dX7 = p["a7"] * X2 - p["b7"] * F5
            dX8 = p["a8"] * F1 - p["b8"] * X8

        return np.array([dX1, dX2, dX3, dX4, dX5, dX6, dX7, dX8], dtype=float)

    # ------------------------ RK4 интегратор --------------------- #

    def _rk4(self, t0: float, t1: float, dt: float, X0: np.ndarray):
        t_values = np.arange(t0, t1 + dt / 2, dt)
        n = len(t_values)
        X = np.zeros((n, 8))
        X[0] = X0

        for i in range(n - 1):
            t = t_values[i]
            x = X[i]
            k1 = dt * self._rhs(t, x)
            k2 = dt * self._rhs(t + dt / 2, x + k1 / 2)
            k3 = dt * self._rhs(t + dt / 2, x + k2 / 2)
            k4 = dt * self._rhs(t + dt, x + k3)
            X[i + 1] = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return t_values, X

    # ------------------------- API simulate ---------------------- #

    def simulate(
        self,
        years: int = 10,
        dt: float = 0.1,
        factor_multipliers: Optional[Dict[str, float]] = None,
        initial_conditions: Optional[np.ndarray] = None,
        equation_coefs: Optional[List[List[float]]] = None,
        **_,
    ):
        if initial_conditions is not None:
            X0 = np.array(initial_conditions, dtype=float)
            if X0.shape[0] != 8:
                raise ValueError("initial_conditions должен содержать 8 значений (X1-X8)")
        else:
            X0 = self.X0.copy()

        self.factor_multipliers = factor_multipliers or {}
        
        # Если переданы коэффициенты уравнений, используем их
        if equation_coefs is not None:
            self.equation_coefs = equation_coefs
            self.use_equations = len(equation_coefs) >= 18
        t_values, X_values = self._rk4(0, years, dt, X0)

        return {
            "time": t_values.tolist(),
            "X1": X_values[:, 0].tolist(),
            "X2": X_values[:, 1].tolist(),
            "X3": X_values[:, 2].tolist(),
            "X4": X_values[:, 3].tolist(),
            "X5": X_values[:, 4].tolist(),
            "X6": X_values[:, 5].tolist(),
            "X7": X_values[:, 6].tolist(),
            "X8": X_values[:, 7].tolist(),
        }
