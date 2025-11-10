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
    X1: List[float]  # нарушения инструкций
    X2: List[float]  # количество катастроф
    X3: List[float]  # коэффициент повторяемости
    X4: List[float]  # доля частных судов
    X5: List[float]  # метеослужбы
    X6: List[float]  # контроль контрафакта
    X7: List[float]  # возраст судов
    X8: List[float]  # квалификация персонала


class VariableInfo(BaseModel):
    """Информация о переменной системы"""
    code: str
    name: str
    description: str
    type: str  # "variable" или "factor"


# Монтирование статических файлов (фронтенд)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


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
        # Переменные системы
        {
            "code": "X1",
            "name": "Нарушения инструкций",
            "description": "Среднее количество нарушений инструкций пилотами",
            "type": "variable"
        },
        {
            "code": "X2",
            "name": "Количество катастроф",
            "description": "Количество авиационных катастроф",
            "type": "variable"
        },
        {
            "code": "X3",
            "name": "Коэффициент повторяемости",
            "description": "Коэффициент повторяемости авиационных происшествий",
            "type": "variable"
        },
        {
            "code": "X4",
            "name": "Доля частных судов",
            "description": "Доля частных судов в авиации",
            "type": "variable"
        },
        {
            "code": "X5",
            "name": "Метеослужбы",
            "description": "Количество сотрудников метеорологических служб",
            "type": "variable"
        },
        {
            "code": "X6",
            "name": "Контроль контрафакта",
            "description": "Показатель активности органов контроля за контрафактом",
            "type": "variable"
        },
        {
            "code": "X7",
            "name": "Возраст судов",
            "description": "Средний возраст воздушных судов",
            "type": "variable"
        },
        {
            "code": "X8",
            "name": "Квалификация персонала",
            "description": "Средняя квалификация персонала",
            "type": "variable"
        },
        # Внешние факторы
        {
            "code": "F1",
            "name": "Лётный стаж",
            "description": "Средний лётный стаж пилотов",
            "type": "factor"
        },
        {
            "code": "F2",
            "name": "Иностранные суда",
            "description": "Доля иностранных воздушных судов",
            "type": "factor"
        },
        {
            "code": "F3",
            "name": "Выработка ресурса",
            "description": "Средняя выработка ресурса до списания",
            "type": "factor"
        },
        {
            "code": "F4",
            "name": "Стоимость топлива",
            "description": "Стоимость авиационного топлива",
            "type": "factor"
        },
        {
            "code": "F5",
            "name": "Контроль государства",
            "description": "Количество нормативно-правовых актов в сфере авиации",
            "type": "factor"
        }
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
    Endpoint для расчета с новой структурой данных
    
    Параметры:
    - startValues: начальные значения переменных X1-X8
    - maxValues: максимальные пределы для X1-X8
    - normValues: нормирующие знаменатели для X1-X8
    - qcoefs: коэффициенты возмущений F1-F5 (полиномы 3-й степени)
    - coefs: коэффициенты уравнений связей (полиномы 3-й степени)
    - years: количество лет для расчёта (по умолчанию 10)
    """
    try:
        # Парсим JSON данные
        start_values = json.loads(startValues)
        max_values = json.loads(maxValues)
        norm_values = json.loads(normValues)
        q_coefs = json.loads(qcoefs)
        equation_coefs = json.loads(coefs)
        
        # Создаем модель и выполняем расчет
        model = AviationSafetyModel()
        
        # Подготавливаем полиномиальные коэффициенты для внешних факторов
        polynomial_coefs = {}
        for i in range(1, 6):
            polynomial_coefs[f'F{i}'] = q_coefs[i-1]
        
        # Запускаем симуляцию с полиномиальными коэффициентами
        results = model.simulate(
            years=years,
            dt=0.1,
            factor_multipliers={},
            initial_conditions=np.array(start_values),
            polynomial_coefs=polynomial_coefs
        )
        
        # Отладочный вывод для проверки значений
        print(f"\n{'='*60}")
        print(f"ОТЛАДОЧНАЯ ИНФОРМАЦИЯ О РАСЧЁТЕ")
        print(f"{'='*60}")
        print(f"Период расчёта: от {results['time'][0]:.2f} до {results['time'][-1]:.2f} лет")
        print(f"Количество точек: {len(results['time'])}")
        print(f"\nРеальные значения переменных (до нормализации):")
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
        
        # Генерируем машинные графики
        image1 = generate_machine_coordinates_linear(results)
        image2 = generate_machine_coordinates_radar(results)
        
        return {
            "image1": image1,  # Линейный график машинных координат от времени
            "image2": image2,  # Объединенные радиальные диаграммы (6 на одном рисунке)
            "status": "success"
        }
        
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


def generate_machine_coordinates_linear(results: Dict) -> str:
    """
    Генерирует линейный график зависимости машинных координат от времени
    
    Args:
        results: словарь с результатами симуляции
        
    Returns:
        base64 строка с изображением
    """
    plt.figure(figsize=(14, 8))
    
    # Используем реальное время (годы)
    time = np.array(results['time'])
    
    # Все переменные системы
    variables = [
        ('X1', 'X1 - Нарушения инструкций', '#1f77b4'),
        ('X2', 'X2 - Количество катастроф', '#ff0000'),
        ('X3', 'X3 - Коэфф. повторяемости', '#ff7f0e'),
        ('X4', 'X4 - Доля частных судов', '#9467bd'),
        ('X5', 'X5 - Метеослужбы', '#2ca02c'),
        ('X6', 'X6 - Контроль контрафакта', '#e377c2'),
        ('X7', 'X7 - Возраст судов', '#7f7f7f'),
        ('X8', 'X8 - Квалификация', '#17becf')
    ]
    
    # Рисуем график для каждой переменной
    for var_key, var_name, color in variables:
        machine_coords = normalize_to_machine_coordinates(results[var_key])
        plt.plot(time, machine_coords, label=var_name, color=color, linewidth=2.5, alpha=0.8)
    
    plt.xlabel('Время (годы)', fontsize=13, fontweight='bold')
    plt.ylabel('Машинные координаты', fontsize=13, fontweight='bold')
    plt.title('Зависимость машинных координат от времени', fontsize=15, fontweight='bold', pad=15)
    
    # Настройка осей
    plt.xlim(time[0], time[-1])  # X от 0 до конца расчётов
    plt.ylim(0, 1)  # Y от 0 до 1
    plt.yticks(np.arange(0, 1.2, 0.2))
    
    # Улучшенная сетка
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    # Легенда в два столбца
    plt.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.9)
    
    plt.tight_layout()
    
    # Конвертируем в base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64


def generate_machine_coordinates_radar(results: Dict) -> str:
    """
    Генерирует одно изображение с 6 радиальными диаграммами для координат 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    
    Args:
        results: словарь с результатами симуляции
        
    Returns:
        base64 строка с изображением
    """
    # Используем реальное время
    time = np.array(results['time'])
    
    # Все переменные системы
    variables = [
        ('X1', 'X1\nНарушения'),
        ('X2', 'X2\nКатастрофы'),
        ('X3', 'X3\nПовтор-ть'),
        ('X4', 'X4\nЧастные'),
        ('X5', 'X5\nМетео'),
        ('X6', 'X6\nКонтроль'),
        ('X7', 'X7\nВозраст'),
        ('X8', 'X8\nКвалиф.')
    ]
    
    # Нормализация всех переменных к машинным координатам
    machine_coords = {}
    for var_key, _ in variables:
        machine_coords[var_key] = normalize_to_machine_coordinates(results[var_key])
    
    # Координаты для радиальных диаграмм
    coord_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Создаем фигуру с 6 подграфиками (2 ряда x 3 колонки)
    fig = plt.figure(figsize=(18, 12))
    
    for idx, coord_val in enumerate(coord_values):
        # Вычисляем реальное время для данной координаты
        target_time = time[0] + coord_val * (time[-1] - time[0])
        
        # Находим ближайший индекс для данного времени
        time_idx = np.argmin(np.abs(time - target_time))
        actual_time = time[time_idx]
        
        # Извлекаем значения всех переменных в этот момент времени
        values = [machine_coords[var_key][time_idx] for var_key, _ in variables]
        labels = [var_name for _, var_name in variables]
        
        # Количество переменных
        num_vars = len(labels)
        
        # Вычисляем углы для каждой оси
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Замыкаем диаграмму
        values += values[:1]
        angles += angles[:1]
        
        # Создаем subplot с полярными координатами
        ax = plt.subplot(2, 3, idx + 1, projection='polar')
        
        # Рисуем диаграмму
        ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4', markersize=6)
        ax.fill(angles, values, alpha=0.25, color='#1f77b4')
        
        # Устанавливаем метки
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9, fontweight='bold')
        
        # Настраиваем радиальную ось от 0 до 1 с шагом 0.2
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        ax.set_yticklabels([f'{val:.1f}' for val in np.arange(0, 1.2, 0.2)], fontsize=7)
        
        # Добавляем сетку
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Заголовок с указанием координаты и времени
        ax.set_title(f'Координата {coord_val:.1f}\n(t = {actual_time:.2f} лет)', 
                     fontsize=11, fontweight='bold', pad=10)
    
    # Общий заголовок для всей фигуры
    fig.suptitle('Радиальные диаграммы для машинных координат 0.0, 0.2, 0.4, 0.6, 0.8, 1.0', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Конвертируем в base64
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

