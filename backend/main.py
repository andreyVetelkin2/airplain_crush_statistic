"""
REST API для веб-сервиса прогнозирования безопасности АТС
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import numpy as np
import os
from pathlib import Path

# Определяем корневую директорию проекта
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

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


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

