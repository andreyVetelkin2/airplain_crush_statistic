// Глобальные переменные
let disastersChart = null;
let allVariablesChart = null;
let comparisonChart = null;
let currentResults = null;
let savedResults = [];

// API базовый URL
const API_BASE = window.location.origin;

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    setupInputListeners();
    
    // Запускаем базовую симуляцию при загрузке
    runSimulation();
});

// Инициализация графиков
function initializeCharts() {
    const ctx1 = document.getElementById('disastersChart').getContext('2d');
    const ctx2 = document.getElementById('allVariablesChart').getContext('2d');
    const ctx3 = document.getElementById('comparisonChart').getContext('2d');
    
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: true,
                position: 'top'
            }
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Время (годы от 2011)'
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Нормированное значение'
                }
            }
        }
    };
    
    disastersChart = new Chart(ctx1, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'X2 - Количество катастроф',
                data: [],
                borderColor: 'rgb(239, 68, 68)',
                backgroundColor: 'rgba(239, 68, 68, 0.1)',
                tension: 0.4,
                fill: true,
                borderWidth: 3
            }]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                title: {
                    display: true,
                    text: 'Прогноз количества авиационных катастроф',
                    font: { size: 16, weight: 'bold' }
                }
            }
        }
    });
    
    allVariablesChart = new Chart(ctx2, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                title: {
                    display: true,
                    text: 'Все переменные системы',
                    font: { size: 16, weight: 'bold' }
                }
            }
        }
    });
    
    comparisonChart = new Chart(ctx3, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                title: {
                    display: true,
                    text: 'Сравнение сценариев (X2 - Количество катастроф)',
                    font: { size: 16, weight: 'bold' }
                }
            }
        }
    });
}

// Настройка слушателей для обновления отображаемых значений
function setupInputListeners() {
    const factorInputs = ['F1', 'F2', 'F3', 'F4', 'F5'];
    factorInputs.forEach(factor => {
        const input = document.getElementById(factor);
        const valueDisplay = input.nextElementSibling;
        
        input.addEventListener('input', function() {
            valueDisplay.textContent = this.value;
        });
    });
}

// Запуск симуляции
async function runSimulation() {
    const statusDiv = document.getElementById('status');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.spinner');
    
    try {
        // Показываем индикатор загрузки
        btnText.textContent = 'Выполняется...';
        spinner.style.display = 'inline-block';
        statusDiv.className = 'status loading';
        statusDiv.textContent = 'Выполняется симуляция...';
        
        // Собираем параметры
        const params = {
            years: parseInt(document.getElementById('years').value),
            dt: parseFloat(document.getElementById('dt').value),
            factor_multipliers: {
                F1: parseFloat(document.getElementById('F1').value),
                F2: parseFloat(document.getElementById('F2').value),
                F3: parseFloat(document.getElementById('F3').value),
                F4: parseFloat(document.getElementById('F4').value),
                F5: parseFloat(document.getElementById('F5').value)
            }
        };
        
        // Отправляем запрос
        const response = await fetch(`${API_BASE}/api/simulate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Ошибка при выполнении симуляции');
        }
        
        const results = await response.json();
        currentResults = results;
        
        // Отладочная информация
        console.log('Параметры симуляции:', params);
        console.log('Результаты симуляции:', results);
        console.log('X2 (катастрофы) начало:', results.X2[0], 'конец:', results.X2[results.X2.length - 1]);
        
        // Обновляем графики
        updateDisastersChart(results);
        updateAllVariablesChart(results);
        
        // Показываем успех
        statusDiv.className = 'status success';
        statusDiv.textContent = '✓ Симуляция выполнена успешно';
        
    } catch (error) {
        console.error('Ошибка:', error);
        statusDiv.className = 'status error';
        statusDiv.textContent = `✗ Ошибка: ${error.message}`;
    } finally {
        btnText.textContent = 'Запустить симуляцию';
        spinner.style.display = 'none';
    }
}

// Обновление графика катастроф
function updateDisastersChart(results) {
    const yearLabels = results.time.map(t => (2011 + t).toFixed(1));
    
    disastersChart.data.labels = yearLabels;
    disastersChart.data.datasets[0].data = results.X2;
    disastersChart.update();
}

// Обновление графика всех переменных
function updateAllVariablesChart(results) {
    const yearLabels = results.time.map(t => (2011 + t).toFixed(1));
    
    const variables = [
        { key: 'X1', label: 'X1 - Нарушения инструкций', color: 'rgb(59, 130, 246)' },
        { key: 'X2', label: 'X2 - Количество катастроф', color: 'rgb(239, 68, 68)' },
        { key: 'X3', label: 'X3 - Коэфф. повторяемости', color: 'rgb(245, 158, 11)' },
        { key: 'X4', label: 'X4 - Доля частных судов', color: 'rgb(139, 92, 246)' },
        { key: 'X5', label: 'X5 - Метеослужбы', color: 'rgb(16, 185, 129)' },
        { key: 'X6', label: 'X6 - Контроль контрафакта', color: 'rgb(236, 72, 153)' },
        { key: 'X7', label: 'X7 - Возраст судов', color: 'rgb(107, 114, 128)' },
        { key: 'X8', label: 'X8 - Квалификация', color: 'rgb(6, 182, 212)' }
    ];
    
    allVariablesChart.data.labels = yearLabels;
    allVariablesChart.data.datasets = variables.map(v => ({
        label: v.label,
        data: results[v.key],
        borderColor: v.color,
        backgroundColor: v.color.replace('rgb', 'rgba').replace(')', ', 0.1)'),
        tension: 0.4,
        borderWidth: 2
    }));
    allVariablesChart.update();
}

// Сохранение результатов для сравнения
function saveCurrentForComparison() {
    if (!currentResults) {
        alert('Сначала запустите симуляцию');
        return;
    }
    
    const params = {
        F1: parseFloat(document.getElementById('F1').value),
        F2: parseFloat(document.getElementById('F2').value),
        F3: parseFloat(document.getElementById('F3').value),
        F4: parseFloat(document.getElementById('F4').value),
        F5: parseFloat(document.getElementById('F5').value)
    };
    
    // Создаем описание сценария
    const scenarioDesc = `Сценарий ${savedResults.length + 1}`;
    
    savedResults.push({
        description: scenarioDesc,
        data: currentResults,
        params: params
    });
    
    updateComparisonChart();
    
    const statusDiv = document.getElementById('status');
    statusDiv.className = 'status success';
    statusDiv.textContent = `✓ Сценарий сохранен для сравнения (всего: ${savedResults.length})`;
}

// Обновление графика сравнения
function updateComparisonChart() {
    if (savedResults.length === 0) {
        comparisonChart.data.labels = [];
        comparisonChart.data.datasets = [];
        comparisonChart.update();
        return;
    }
    
    const yearLabels = savedResults[0].data.time.map(t => (2011 + t).toFixed(1));
    
    const colors = [
        'rgb(59, 130, 246)',
        'rgb(239, 68, 68)',
        'rgb(16, 185, 129)',
        'rgb(245, 158, 11)',
        'rgb(139, 92, 246)',
        'rgb(236, 72, 153)'
    ];
    
    comparisonChart.data.labels = yearLabels;
    comparisonChart.data.datasets = savedResults.map((result, index) => ({
        label: result.description,
        data: result.data.X2,
        borderColor: colors[index % colors.length],
        backgroundColor: colors[index % colors.length].replace('rgb', 'rgba').replace(')', ', 0.1)'),
        tension: 0.4,
        borderWidth: 2
    }));
    comparisonChart.update();
}

// Очистка сравнения
function clearComparison() {
    savedResults = [];
    updateComparisonChart();
    
    const statusDiv = document.getElementById('status');
    statusDiv.className = 'status success';
    statusDiv.textContent = '✓ Сравнение очищено';
}

// Переключение табов
function switchTab(tabName) {
    // Скрываем все табы
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Показываем нужный таб
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');
}

// Предустановленные сценарии
function setScenario(scenarioType) {
    switch(scenarioType) {
        case 'base':
            // Базовый сценарий
            document.getElementById('F1').value = 1.0;
            document.getElementById('F2').value = 1.0;
            document.getElementById('F3').value = 1.0;
            document.getElementById('F4').value = 1.0;
            document.getElementById('F5').value = 1.0;
            break;
            
        case 'control_increase':
            // Увеличение контроля со стороны государства
            document.getElementById('F1').value = 1.0;
            document.getElementById('F2').value = 1.0;
            document.getElementById('F3').value = 1.0;
            document.getElementById('F4').value = 1.0;
            document.getElementById('F5').value = 1.5; // Увеличение НПА
            break;
            
        case 'repeat_increase':
            // Увеличение повторяемости (косвенно через начальные условия)
            // В данном случае уменьшаем контроль, что приведет к росту повторяемости
            document.getElementById('F1').value = 0.8;
            document.getElementById('F2').value = 1.2;
            document.getElementById('F3').value = 1.0;
            document.getElementById('F4').value = 1.3;
            document.getElementById('F5').value = 0.7;
            break;
    }
    
    // Обновляем отображаемые значения
    ['F1', 'F2', 'F3', 'F4', 'F5'].forEach(factor => {
        const input = document.getElementById(factor);
        const valueDisplay = input.nextElementSibling;
        valueDisplay.textContent = input.value;
    });
    
    const statusDiv = document.getElementById('status');
    statusDiv.className = 'status success';
    statusDiv.textContent = `✓ Установлен сценарий: ${getScenarioName(scenarioType)}`;
}

function getScenarioName(scenarioType) {
    const names = {
        'base': 'Базовый',
        'control_increase': 'Увеличенный контроль государства',
        'repeat_increase': 'Увеличенная повторяемость'
    };
    return names[scenarioType] || scenarioType;
}

// Сброс параметров
function resetParams() {
    document.getElementById('years').value = 10;
    document.getElementById('dt').value = 0.1;
    setScenario('base');
    
    const statusDiv = document.getElementById('status');
    statusDiv.className = 'status success';
    statusDiv.textContent = '✓ Параметры сброшены';
}

