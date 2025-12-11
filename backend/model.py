# backend/model.py
import numpy as np
from typing import Dict, Optional, Tuple, List


class AviationSafetyModel:

    _MAX_BASE_FOR_POW = 10.0

    _COEF_LIMITS = {
        "a": (-0.5, 0.5),
        "b": (-0.5, 0.5),
        "c": (-1.0, 1.0),
        "d": (-1.0, 1.0),
    }

    _MAX_DX_PER_STEP = 5.0

    def __init__(self, params: Optional[Dict[str, float]] = None, equation_coefs: Optional[List[List[float]]] = None):
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

        self.equation_coefs = self._sanitize_equation_coefs(equation_coefs)
        self.use_equations = self.equation_coefs is not None and len(self.equation_coefs) >= 18

    def _clip_factor(self, name: str, value: float) -> float:
        low, high = self.factor_bounds[name]
        return float(np.clip(value, low, high))

    def _sanitize_equation_coefs(self, coefs: Optional[List[List[float]]]) -> Optional[List[List[float]]]:
        if coefs is None:
            return None
        safe = []
        for idx, quad in enumerate(coefs):
            a = quad[0] if len(quad) > 0 else 0.0
            b = quad[1] if len(quad) > 1 else 0.0
            c = quad[2] if len(quad) > 2 else 0.0
            d = quad[3] if len(quad) > 3 else 0.0
            a = float(np.clip(a, *self._COEF_LIMITS["a"]))
            b = float(np.clip(b, *self._COEF_LIMITS["b"]))
            c = float(np.clip(c, *self._COEF_LIMITS["c"]))
            d = float(np.clip(d, *self._COEF_LIMITS["d"]))
            safe.append([a, b, c, d])
        return safe

    def _clip_coefs_inplace(self, coefs: List[List[float]]):
        for i, quad in enumerate(coefs):
            for j, name in enumerate(["a", "b", "c", "d"]):
                low, high = self._COEF_LIMITS[name]
                coefs[i][j] = float(np.clip(quad[j], low, high))

    def _prepare_x_for_pow(self, x: float) -> float:
        x_t = np.tanh(x)
        x_t = float(np.clip(x_t, -self._MAX_BASE_FOR_POW, self._MAX_BASE_FOR_POW))
        return x_t

    def _safe_pow(self, base: float, power: int) -> float:
        b = self._prepare_x_for_pow(base)
        with np.errstate(over="ignore", invalid="ignore"):
            try:
                if power == 3:
                    res = b * b * b
                elif power == 2:
                    res = b * b
                else:
                    res = b ** power
            except Exception:
                res = 0.0
        res = float(np.nan_to_num(res, nan=0.0, posinf=np.sign(res) * 1e6, neginf=np.sign(res) * -1e6))
        return res


    def _compute_polynomial(self, coefs: List[float], x: float) -> float:
        if coefs is None or len(coefs) < 4:
            return 0.0
        # защита коэффициентов
        a = float(np.clip(coefs[0], *self._COEF_LIMITS["a"]))
        b = float(np.clip(coefs[1], *self._COEF_LIMITS["b"]))
        c = float(np.clip(coefs[2], *self._COEF_LIMITS["c"]))
        d = float(np.clip(coefs[3], *self._COEF_LIMITS["d"]))

        x3 = self._safe_pow(x, 3)
        x2 = self._safe_pow(x, 2)

        with np.errstate(over="ignore", invalid="ignore"):
            val = a * x3 + b * x2 + c * x + d
        val = float(np.nan_to_num(val, nan=0.0, posinf=1e6, neginf=-1e6))
        return val

    def _compute_polynomial_from_var(self, func_num: int, var_value: float, X: np.ndarray) -> float:
        if not self.use_equations or func_num < 1 or func_num > len(self.equation_coefs):
            return 0.0
        coefs = self.equation_coefs[func_num - 1]
        return self._compute_polynomial(coefs, var_value)

    def _compute_f(self, i: int, X: np.ndarray) -> float:
        if not self.use_equations or i < 1 or i > len(self.equation_coefs):
            return 0.0

        coefs = self.equation_coefs[i - 1]

        if 1 <= i <= 8:
            x = X[i - 1]
            return self._compute_polynomial(coefs, x)

        try:
            if i == 9:
                return (coefs[0] * self._safe_pow(X[0], 3)
                        + coefs[1] * self._safe_pow(X[1], 2)
                        + coefs[2] * X[2]
                        + coefs[3])
            elif i == 10:
                return (coefs[0] * self._safe_pow(X[3], 3)
                        + coefs[1] * self._safe_pow(X[4], 2)
                        + coefs[2] * X[5]
                        + coefs[3])
            elif i == 11:
                return (coefs[0] * self._safe_pow(X[6], 3)
                        + coefs[1] * self._safe_pow(X[7], 2)
                        + coefs[2] * X[0]
                        + coefs[3])
            elif i == 12:
                return (coefs[0] * self._safe_pow(X[1], 3)
                        + coefs[1] * self._safe_pow(X[2], 2)
                        + coefs[2] * X[3]
                        + coefs[3])
            elif i == 13:
                return (coefs[0] * self._safe_pow(X[4], 3)
                        + coefs[1] * self._safe_pow(X[5], 2)
                        + coefs[2] * X[6]
                        + coefs[3])
            elif i == 14:
                return (coefs[0] * self._safe_pow(X[7], 3)
                        + coefs[1] * self._safe_pow(X[0], 2)
                        + coefs[2] * X[1]
                        + coefs[3])
            elif i == 15:
                return (coefs[0] * self._safe_pow(X[2], 3)
                        + coefs[1] * self._safe_pow(X[3], 2)
                        + coefs[2] * X[4]
                        + coefs[3])
            elif i == 16:
                return (coefs[0] * self._safe_pow(X[5], 3)
                        + coefs[1] * self._safe_pow(X[6], 2)
                        + coefs[2] * X[7]
                        + coefs[3])
            elif i == 17:
                return (coefs[0] * self._safe_pow(X[0], 3)
                        + coefs[1] * self._safe_pow(X[2], 2)
                        + coefs[2] * X[4]
                        + coefs[3])
            elif i == 18:
                return (coefs[0] * self._safe_pow(X[1], 3)
                        + coefs[1] * self._safe_pow(X[3], 2)
                        + coefs[2] * X[5]
                        + coefs[3])
            elif i == 19:
                return (coefs[0] * self._safe_pow(X[6], 3)
                        + coefs[1] * self._safe_pow(X[0], 2)
                        + coefs[2] * X[7]
                        + coefs[3])
            elif i == 20:
                return (coefs[0] * self._safe_pow(X[3], 3)
                        + coefs[1] * self._safe_pow(X[1], 2)
                        + coefs[2] * X[2]
                        + coefs[3])
        except Exception:
            return 0.0

        return 0.0


    def _rhs(self, t: float, X: np.ndarray) -> np.ndarray:
        X1, X2, X3, X4, X5, X6, X7, X8 = X
        F1, F2, F3, F4, F5 = self._factors(t)

        if self.use_equations:
            f1_X2 = self._compute_polynomial_from_var(1, X2, X)
            f2_X3 = self._compute_polynomial_from_var(2, X3, X)
            f3_X4 = self._compute_polynomial_from_var(3, X4, X)
            f4_X4 = self._compute_polynomial_from_var(4, X4, X)
            f5_X6 = self._compute_polynomial_from_var(5, X6, X)
            f6_X7 = self._compute_polynomial_from_var(6, X7, X)
            f7_X8 = self._compute_polynomial_from_var(7, X8, X)
            f8_X7 = self._compute_polynomial_from_var(8, X7, X)

            f9_val = self._compute_f(9, X)
            f10_val = self._compute_f(10, X)
            f11_val = self._compute_f(11, X)
            f12_val = self._compute_f(12, X)
            f13_val = self._compute_f(13, X)
            f14_val = self._compute_f(14, X)
            f15_val = self._compute_f(15, X)
            f16_val = self._compute_f(16, X)
            f17_val = self._compute_f(17, X)
            f18_val = self._compute_f(18, X)

            dX1 = F3 - f1_X2 * f2_X3 * f3_X4
            dX2 = (F1 + F2 + F4) * f4_X4 * f5_X6 * f6_X7 - (F3 + F5) * f7_X8
            dX3 = F4 * f8_X7 - (F3 + F5) * f9_val
            dX4 = F2 * f10_val * f11_val - (F3 + F5) * f12_val
            f13_X2 = self._compute_polynomial_from_var(13, X2, X) if self.equation_coefs is not None else f13_val
            f14_X2 = self._compute_polynomial_from_var(14, X2, X) if self.equation_coefs is not None else f14_val

            dX5 = f13_X2 - F5
            dX6 = F2 * f14_X2 - F5
            dX7 = F2 * f15_val * f16_val * f17_val - (F3 + F5)
            dX8 = F5 - (self._compute_polynomial_from_var(18, X2, X) if self.equation_coefs is not None else f18_val)

        else:
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

        d = np.array([dX1, dX2, dX3, dX4, dX5, dX6, dX7, dX8], dtype=float)
        d = np.nan_to_num(d, nan=0.0, posinf=1e6, neginf=-1e6)

        max_dx = self._MAX_DX_PER_STEP
        d = np.sign(d) * np.minimum(np.abs(d), max_dx)

        return d


    def _rk4(self, t0: float, t1: float, dt: float, X0: np.ndarray):
        t_values = np.arange(t0, t1 + dt / 2, dt)
        n = len(t_values)
        X = np.zeros((n, 8))
        X[0] = X0

        for i in range(n - 1):
            t = t_values[i]
            x = X[i].copy()

            k1 = dt * self._rhs(t, x)
            k2 = dt * self._rhs(t + dt / 2, x + k1 / 2)
            k3 = dt * self._rhs(t + dt / 2, x + k2 / 2)
            k4 = dt * self._rhs(t + dt, x + k3)

            next_x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

            if np.any(~np.isfinite(next_x)):
                finite_mask = np.isfinite(next_x)
                next_x[~finite_mask] = x[~finite_mask]
                next_x = x + 0.5 * (next_x - x)

            next_x = np.clip(next_x, -10.0, 10.0)

            X[i + 1] = next_x

        return t_values, X

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

        if equation_coefs is not None:
            self.equation_coefs = self._sanitize_equation_coefs(equation_coefs)
            self.use_equations = len(self.equation_coefs) >= 18

        if dt > 0.5:
            dt = 0.5
        if dt <= 0:
            dt = 0.1

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

    def _factors(self, t: float, factor_multipliers: Optional[Dict[str, float]] = None):
        multipliers = factor_multipliers or self.factor_multipliers or {}

        base_F1 = 0.63 + 0.37 * t
        base_F2 = 1.0 - 0.23 * t
        base_F3 = 1.0 - 0.33 * t
        base_F4 = 0.51 + 0.46 * t
        base_F5 = 0.6 + 0.4 * t

        F1 = self._clip_factor("F1", base_F1 * multipliers.get("F1", 1.0))
        F2 = self._clip_factor("F2", base_F2 * multipliers.get("F2", 1.0))
        F3 = self._clip_factor("F3", base_F3 * multipliers.get("F3", 1.0))
        F4 = self._clip_factor("F4", base_F4 * multipliers.get("F4", 1.0))
        F5 = self._clip_factor("F5", base_F5 * multipliers.get("F5", 1.0))

        return F1, F2, F3, F4, F5