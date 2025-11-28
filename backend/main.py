"""
REST API для веб-сервиса прогнозирования безопасности АТС
"""

from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import numpy as np
import os
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Используем non-GUI бэкенд
import matplotlib.pyplot as plt
import io
import base64
import logging

# Определяем корневую директорию проекта
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Добавляем путь к корню проекта для импортов
import sys
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.model import AviationSafetyModel
except ImportError:
    from model import AviationSafetyModel

app = FastAPI(
    title="Aviation Safety Forecasting System",
    description="Система прогнозирования характеристик безопасности авиационных транспортных систем",
    version="1.0.0"
)

# CORS middleware для возможности обращения с фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationRequest(BaseModel):
    """Параметры запроса на симуляцию"""
    years: int = Field(default=10, ge=1, le=50, description="Количество лет для прогноза")
    dt: float = Field(default=0.1, gt=0, le=1, description="Шаг интегрирования")
    factor_multipliers: Optional[Dict[str, float]] = Field(
        default=None,
        description="Множители для внешних факторов (F1-F5)"
    )
    initial_conditions: Optional[List[float]] = Field(
        default=None,
        description="Начальные условия для переменных X1-X8"
    )


class SimulationResponse(BaseModel):
    """Результаты симуляции"""
    time: List[float]
    X1: List[float]
    X2: List[float]
    X3: List[float]
    X4: List[float]
    X5: List[float]
    X6: List[float]
    X7: List[float]
    X8: List[float]


class VariableInfo(BaseModel):
    """Информация о переменной системы"""
    code: str
    name: str
    description: str
    type: str  # "variable" или "factor"


# Монтирование статических файлов (фронтенд)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


logger = logging.getLogger("aviation.safety")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@app.get("/")
async def root():
    """Главная страница - возвращает фронтенд"""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Aviation Safety Forecasting System API", "docs": "/docs"}


@app.get("/api/variables", response_model=List[VariableInfo])
async def get_variables():
    """Получить список всех переменных и внешних факторов системы"""
    return [
        {"code": "X1", "name": "Нарушения пилотов", "description": "Среднее количество нарушений инструкций", "type": "variable"},
        {"code": "X2", "name": "Катастрофы", "description": "Количество авиационных катастроф", "type": "variable"},
        {"code": "X3", "name": "Повторяемость", "description": "Коэффициент повторяемости происшествий", "type": "variable"},
        {"code": "X4", "name": "Частные суда", "description": "Доля частных судов в перевозках", "type": "variable"},
        {"code": "X5", "name": "Метеослужбы", "description": "Количество сотрудников метеорологических служб", "type": "variable"},
        {"code": "X6", "name": "Контроль контрафакта", "description": "Активность контрольных органов", "type": "variable"},
        {"code": "X7", "name": "Возраст судов", "description": "Средний возраст флота", "type": "variable"},
        {"code": "X8", "name": "Квалификация персонала", "description": "Средний уровень подготовки", "type": "variable"},
        {"code": "F1", "name": "Контроль государства", "description": "Нормативно-правовые акты", "type": "factor"},
        {"code": "F2", "name": "Инфраструктура", "description": "Топливо, снабжение, логистика", "type": "factor"},
        {"code": "F3", "name": "Условия среды", "description": "Природно-климатические условия", "type": "factor"},
        {"code": "F4", "name": "Иностранные самолёты", "description": "Доля иностранных судов", "type": "factor"},
        {"code": "F5", "name": "Законодательство", "description": "Регуляторные ограничения", "type": "factor"},
    ]


@app.post("/api/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """
    Запустить симуляцию модели
    
    Параметры:
    - years: количество лет для прогноза (1-50)
    - dt: шаг интегрирования (0.01-1.0)
    - factor_multipliers: множители для внешних факторов F1-F5
    - initial_conditions: начальные условия для X1-X8
    """
    try:
        model = AviationSafetyModel()
        
        # Преобразование начальных условий
        initial_cond = None
        if request.initial_conditions is not None:
            if len(request.initial_conditions) != 8:
                raise HTTPException(
                    status_code=400,
                    detail="initial_conditions должен содержать ровно 8 значений (X1-X8)"
                )
            initial_cond = np.array(request.initial_conditions)
        
        # Запуск симуляции
        results = model.simulate(
            years=request.years,
            dt=request.dt,
            factor_multipliers=request.factor_multipliers or {},
            initial_conditions=initial_cond
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при выполнении симуляции: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Проверка работоспособности API"""
    return {"status": "ok", "message": "API работает нормально"}


def plot_dtp_to_base64(results: Dict) -> str:
    """
    Использует логику plot_dtp из test.py для генерации графика.
    Возвращает base64 строку изображения.
    """
    base_year = 2011
    time_values = base_year + np.array(results['time'])
    
    variable_names = {
        'X1': 'X1 - Нарушения',
        'X2': 'X2 - Катастрофы',
        'X3': 'X3 - Повторяемость',
        'X4': 'X4 - Частные суда',
        'X5': 'X5 - Метеослужбы',
        'X6': 'X6 - Контроль контрафакта',
        'X7': 'X7 - Возраст судов',
        'X8': 'X8 - Квалификация'
    }
    
    data = {}
    for var in ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']:
        data[variable_names[var]] = results[var]
    
    years = time_values.tolist()
    
    plt.figure(figsize=(14, 8))
    plt.grid(True, alpha=0.3)
    
    for label, values in data.items():
        plt.plot(years, values, marker='o', linewidth=2, label=label)
    
    plt.xlabel("Годы")
    plt.ylabel("Значения основных показателей безопасности ДТП")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64


@app.post("/calculate/")
async def calculate(
    startValues: str = Form(...),
    maxValues: str = Form(...),
    normValues: str = Form(...),
    qcoefs: str = Form(...),
    coefs: str = Form(...),
    years: int = Form(10)
):
    """
    Быстрый расчёт модели с использованием уравнений связей переменных.
    Поля maxValues/normValues оставлены для обратной совместимости и не используются.
    """
    try:
        start_values_raw = json.loads(startValues)
        if len(start_values_raw) < 8:
            raise HTTPException(status_code=400, detail="startValues должен содержать минимум 8 значений (X1-X8)")
        initial = np.array(start_values_raw[:8], dtype=float)

        factor_multipliers: Dict[str, float] = {}
        try:
            q_data = json.loads(qcoefs)
            if isinstance(q_data, dict):
                for code in ("F1", "F2", "F3", "F4", "F5"):
                    if code in q_data:
                        factor_multipliers[code] = float(q_data[code])
        except Exception:
            pass

        # Парсим коэффициенты уравнений f1-f20
        equation_coefs = None
        try:
            coefs_data = json.loads(coefs)
            if isinstance(coefs_data, list) and len(coefs_data) >= 20:
                # Проверяем, что каждый элемент - список из 4 коэффициентов
                equation_coefs = []
                for i, coef_row in enumerate(coefs_data[:20]):
                    if isinstance(coef_row, list) and len(coef_row) >= 4:
                        equation_coefs.append([float(coef_row[0]), float(coef_row[1]), 
                                             float(coef_row[2]), float(coef_row[3])])
                    else:
                        # Если формат некорректный, используем значения по умолчанию (0)
                        equation_coefs.append([0.0, 0.0, 0.0, 0.0])
                if len(equation_coefs) == 20:
                    logger.info(f"Загружены коэффициенты для {len(equation_coefs)} уравнений")
        except Exception as e:
            logger.warning(f"Не удалось распарсить коэффициенты уравнений: {e}. Используется старая модель.")

        model = AviationSafetyModel(equation_coefs=equation_coefs)
        results = model.simulate(
            years=years,
            dt=0.1,
            factor_multipliers=factor_multipliers,
            initial_conditions=initial,
            equation_coefs=equation_coefs,
        )

        validate_simulation_results(results)

        print(f"\n{'='*60}")
        print("ОТЛАДОЧНАЯ ИНФОРМАЦИЯ О РАСЧЁТЕ")
        print(f"{'='*60}")
        print(f"Период расчёта: от {results['time'][0]:.2f} до {results['time'][-1]:.2f} лет")
        print(f"Количество точек: {len(results['time'])}")
        print("\nРеальные значения переменных (до нормализации):")
        print(f"{'Переменная':<12} {'Минимум':<12} {'Максимум':<12} {'Начало':<12} {'Конец':<12} {'Изменение'}")
        print(f"{'-'*60}")

        for var in ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']:
            values = results[var]
            min_val = min(values)
            max_val = max(values)
            start_val = values[0]
            end_val = values[-1]
            change = end_val - start_val
            print(f"{var:<12} {min_val:<12.4f} {max_val:<12.4f} {start_val:<12.4f} {end_val:<12.4f} {change:+.4f}")

        print(f"{'='*60}\n")

        image1 = plot_dtp_to_base64(results)
        image2 = generate_machine_coordinates_radar(results)

        return {
            "image1": image1,
            "image2": image2,
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при выполнении расчета: {str(e)}")


def normalize_to_machine_coordinates(values):
    """
    Нормализует значения к машинным координатам [0, 1]
    
    Args:
        values: массив значений
        
    Returns:
        нормализованный массив
    """
    values = np.array(values)
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val - min_val < 1e-10:
        return np.ones_like(values) * 0.5
    return (values - min_val) / (max_val - min_val)


def validate_simulation_results(results: Dict) -> None:
    """
    Проверяет результаты симуляции на наличие некорректных значений.

    Если в массивах X1–X8 присутствуют NaN/Inf или значения
    выходят далеко за разумные пределы, генерируется HTTPException.

    Это помогает вовремя отловить ситуации, когда из‑за слишком
    «жёстких» коэффициентов (например, полиномов F1–F5) решение
    системы дифференциальных уравнений численно «разлетается»,
    а графики оказываются визуально пустыми.
    """
    from fastapi import HTTPException

    bad_series = []
    for key in ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]:
        arr = np.array(results.get(key, []), dtype=float)
        if arr.size == 0:
            bad_series.append(f"{key}: пустой ряд значений")
            continue

        # Проверка на NaN/Inf
        if not np.all(np.isfinite(arr)):
            bad_series.append(f"{key}: присутствуют NaN/Inf")
            continue

        # Проверка на «слишком большие» значения (явный численный взрыв)
        if np.any(np.abs(arr) > 1e6):
            bad_series.append(f"{key}: значения выходят за разумные пределы (>|1e6|)")

    if bad_series:
        details = ";\n".join(bad_series)
        raise HTTPException(
            status_code=400,
            detail=(
                "Результат расчёта содержит некорректные значения и не может быть "
                "правильно отображён на графиках.\n"
                "Скорректируйте коэффициенты (особенно полиномы F1–F5) или уменьшите "
                "горизонт моделирования.\n"
                f"Диагностика по рядам: {details}"
            ),
        )


def _log_series_debug(series_name: str, time: np.ndarray, values: np.ndarray) -> None:
    """
    Выводит подробный лог по временным рядам, чтобы отладить «падение в ноль».
    """
    if time.size == 0 or values.size == 0:
        logger.warning(
            "[plot-debug] %s: пустые массивы (time=%d, values=%d)",
            series_name,
            time.size,
            values.size,
        )
        return

    logger.info(
        "[plot-debug] %s: n=%d, time=[%.2f..%.2f], value=[%.4f..%.4f], "
        "min=%.4f, max=%.4f",
        series_name,
        len(values),
        time[0],
        time[-1],
        values[0],
        values[-1],
        values.min(),
        values.max(),
    )

    tail = values[-5:]
    tail_times = time[-5:]
    logger.info(
        "[plot-debug] %s tail: %s",
        series_name,
        ", ".join(
            f"(t={t:.2f}, x={x:.4f})" for t, x in zip(tail_times, tail)
        ),
    )


def generate_machine_coordinates_linear(results: Dict) -> str:
    """
    Основной график: годы по X, катастрофы по левому Y, остальные переменные нормированы и выведены по правому Y.
    """
    plt.figure(figsize=(14, 8))

    base_year = 2011
    time = base_year + np.array(results['time'])
    series = {key: np.array(results[key]) for key in ['X1','X2','X3','X4','X5','X6','X7','X8']}

    catastrophes = series['X2']
    _log_series_debug("X2-catastrophes", time, catastrophes)

    ax_cat = plt.gca()
    ax_rest = ax_cat.twinx()

    ax_cat.plot(
        time,
        normalize_to_machine_coordinates(catastrophes),
        color='#d62728',
        linewidth=2.2,
        linestyle='-',
        marker=None,
        label='X2 – Катастрофы (норм.)'
    )
    ax_cat.set_ylim(0, 1.05)
    ax_cat.set_ylabel('Катастрофы (X2) [0..1]', fontsize=13, fontweight='bold', color='#d62728')
    ax_cat.tick_params(axis='y', labelcolor='#d62728')

    variables = [
        ('X1', 'Нарушения', '#1f77b4'),
        ('X3', 'Повторяемость', '#ff7f0e'),
        ('X4', 'Частные суда', '#9467bd'),
        ('X5', 'Метеослужбы', '#2ca02c'),
        ('X6', 'Контроль контрафакта', '#e377c2'),
        ('X7', 'Возраст судов', '#7f7f7f'),
        ('X8', 'Квалификация', '#17becf'),
    ]

    norm_series = {key: normalize_to_machine_coordinates(series[key]) for key in series}
    for key, title, color in variables:
        ax_rest.plot(
            time,
            norm_series[key],
            color=color,
            linewidth=1.8,
            alpha=0.85,
            label=f"{key} – {title}"
        )

    ax_rest.set_ylim(0, 1.05)
    ax_rest.set_ylabel('Нормированные значения (X1, X3–X8)', fontsize=13, fontweight='bold')

    ax_cat.set_xlabel('Годы (2011+)', fontsize=13, fontweight='bold')
    ax_cat.set_title('Показатели безопасности X1–X8', fontsize=16, fontweight='bold', pad=15)

    ax_cat.set_xlim(time[0], time[-1])
    years_range = np.arange(int(time[0]), int(time[-1]) + 1)
    if len(years_range) <= 20:
        ax_cat.set_xticks(years_range)
    else:
        step = max(1, len(years_range) // 12)
        ax_cat.set_xticks(years_range[::step])
    ax_cat.set_xticklabels(ax_cat.get_xticks(), rotation=30)

    ax_cat.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)

    lines, labels = ax_cat.get_legend_handles_labels()
    lines2, labels2 = ax_rest.get_legend_handles_labels()
    ax_rest.legend(
        lines + lines2,
        labels + labels2,
        loc='upper left',
        fontsize=9,
        framealpha=0.9,
        ncol=2
    )

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()

    return image_base64


def generate_machine_coordinates_radar(results: Dict) -> str:
    """
    Снимки «круговых» диаграмм: для шести временных координат показываем профиль X1–X4.
    """
    base_year = 2011
    time = base_year + np.array(results['time'])

    variables = [
        ('X1', 'Нарушения'),
        ('X2', 'Катастрофы'),
        ('X3', 'Повторы'),
        ('X4', 'Частные'),
        ('X5', 'Метео'),
        ('X6', 'Контроль'),
        ('X7', 'Возраст'),
        ('X8', 'Квалиф.'),
    ]

    norm_series = {code: normalize_to_machine_coordinates(results[code]) for code, _ in variables}

    coord_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    fig = plt.figure(figsize=(18, 12))

    for idx, coord in enumerate(coord_values):
        target_time = time[0] + coord * (time[-1] - time[0])
        time_idx = np.argmin(np.abs(time - target_time))
        actual_time = time[time_idx]

        values = [norm_series[code][time_idx] for code, _ in variables]
        labels = [label for _, label in variables]
        angles = np.linspace(0, 2 * np.pi, len(variables), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax = plt.subplot(2, 3, idx + 1, projection='polar')
        ax.plot(angles, values, linewidth=2.2, color='#ff7f0e')
        ax.fill(angles, values, alpha=0.25, color='#ff7f0e')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0.0, 1.1, 0.2))
        ax.set_yticklabels([f'{v:.1f}' for v in np.arange(0.0, 1.1, 0.2)], fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(f'Координата {coord:.1f}\nГод {actual_time:.1f}', fontsize=11, fontweight='bold')

    fig.suptitle('Круговые диаграммы X1–X8 (нормированные значения)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()

    return image_base64




if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

