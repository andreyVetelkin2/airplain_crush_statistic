// Константы
const NUM_VARIABLES = 8;  // X1-X8
const NUM_DISTURBANCES = 5;  // F1-F5
const NUM_EQUATIONS = 20;  // f1-f20

// Инициализация при загрузке страницы
document.addEventListener("DOMContentLoaded", function() {
    clearPreviousResults();
    setupEventListeners();
});

// Настройка обработчиков событий
function setupEventListeners() {
    document.getElementById("random").addEventListener("click", randomizeValues);
    document.getElementById("saveValuesToJson").addEventListener("click", saveValuesToJson);
    document.getElementById("loadValuesFromJson").addEventListener("click", loadValuesFromJson);
    document.getElementById("calculate").addEventListener("click", calculate);
    setupImageZoom();
}

// Функция случайного заполнения
function randomizeValues() {
    const probInput = parseFloat(document.getElementById("chanceInput").value);
    const probNonZero = (isNaN(probInput) ? 0.9 : Math.min(Math.max(probInput, 0), 1));

    const MIN_NONZERO = 0.05;

    function randUnit() {
        return Math.random();
    }

    function randNonZero() {
        return MIN_NONZERO + (1 - MIN_NONZERO) * Math.random();
    }

    function getConditionalRandom() {
        return Math.random() < probNonZero ? randNonZero() : 0.0;
    }

    for (let i = 1; i <= NUM_VARIABLES; i++) {
        const val = randNonZero();
        // показываем с двумя знаками в UI, но числовая генерация остаётся корректной
        document.getElementById(`X${i}`).value = val.toFixed(2);
        document.getElementById(`X${i}_max`).value = (val * 1.25).toFixed(2);
    }

    for (let i = 1; i <= NUM_DISTURBANCES; i++) {
        document.getElementById(`qa${i}`).value = getConditionalRandom().toFixed(2);
        document.getElementById(`qb${i}`).value = getConditionalRandom().toFixed(2);
        document.getElementById(`qc${i}`).value = getConditionalRandom().toFixed(2);
        document.getElementById(`qd${i}`).value = getConditionalRandom().toFixed(2);
    }

    for (let i = 1; i <= NUM_EQUATIONS; i++) {
        // можно чуть увеличить MIN_NONZERO для коэффициентов уравнений, но для простоты используем getConditionalRandom()
        document.getElementById(`a${i}`).value = getConditionalRandom().toFixed(2);
        document.getElementById(`b${i}`).value = getConditionalRandom().toFixed(2);
        document.getElementById(`c${i}`).value = getConditionalRandom().toFixed(2);
        document.getElementById(`d${i}`).value = getConditionalRandom().toFixed(2);
    }
}

// Функция сохранения в JSON
function saveValuesToJson() {
    // Собираем значения X1-X8 (startValues)
    const startValues = [];
    for (let i = 1; i <= NUM_VARIABLES; i++) {
        const value = parseFloat(document.getElementById(`X${i}`).value) || 0;
        startValues.push(value);
    }

    // Собираем значения X1-X8 MAX (maxValues)
    const maxValues = [];
    for (let i = 1; i <= NUM_VARIABLES; i++) {
        const value = parseFloat(document.getElementById(`X${i}_max`).value) || 0;
        maxValues.push(value);
    }

    // Собираем значения X1-X8 norm (normValues)
    const normValues = [];
    for (let i = 1; i <= NUM_VARIABLES; i++) {
        const value = parseFloat(document.getElementById(`X${i}_norm`).value) || 1.0;
        normValues.push(value);
    }

    // Собираем коэффициенты возмущений
    const qcoefs = [];
    for (let i = 1; i <= NUM_DISTURBANCES; i++) {
        const qa = parseFloat(document.getElementById(`qa${i}`).value) || 0;
        const qb = parseFloat(document.getElementById(`qb${i}`).value) || 0;
        const qc = parseFloat(document.getElementById(`qc${i}`).value) || 0;
        const qd = parseFloat(document.getElementById(`qd${i}`).value) || 0;
        qcoefs.push([qa, qb, qc, qd]);
    }

    // Собираем коэффициенты уравнений
    const coefs = [];
    for (let i = 1; i <= NUM_EQUATIONS; i++) {
        const a = parseFloat(document.getElementById(`a${i}`).value) || 0;
        const b = parseFloat(document.getElementById(`b${i}`).value) || 0;
        const c = parseFloat(document.getElementById(`c${i}`).value) || 0;
        const d = parseFloat(document.getElementById(`d${i}`).value) || 0;
        coefs.push([a, b, c, d]);
    }

    // Создаем объект для сохранения
    const data = {
        startValues,
        maxValues,
        normValues,
        qcoefs,
        coefs
    };

    // Преобразуем объект в JSON строку
    const jsonData = JSON.stringify(data, null, 2);

    // Создаем Blob и ссылку для скачивания
    const blob = new Blob([jsonData], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    // Создаем временную ссылку и инициируем скачивание
    const a = document.createElement("a");
    a.href = url;
    a.download = "aviation_safety_data.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Функция загрузки из JSON
function loadValuesFromJson() {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.json';
    fileInput.onchange = loadFromJson;
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);
    fileInput.click();
}

function loadFromJson(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const jsonData = JSON.parse(e.target.result);

            // Заполнение значений переменных
            if (jsonData.startValues) {
                jsonData.startValues.forEach((value, index) => {
                    document.getElementById(`X${index + 1}`).value = value;
                });
            }

            if (jsonData.maxValues) {
                jsonData.maxValues.forEach((value, index) => {
                    document.getElementById(`X${index + 1}_max`).value = value;
                });
            }

            if (jsonData.normValues) {
                jsonData.normValues.forEach((value, index) => {
                    document.getElementById(`X${index + 1}_norm`).value = value;
                });
            }

            // Заполнение коэффициентов возмущений
            if (jsonData.qcoefs) {
                jsonData.qcoefs.forEach((coef, index) => {
                    document.getElementById(`qa${index + 1}`).value = coef[0];
                    document.getElementById(`qb${index + 1}`).value = coef[1];
                    document.getElementById(`qc${index + 1}`).value = coef[2];
                    document.getElementById(`qd${index + 1}`).value = coef[3];
                });
            }

            // Заполнение коэффициентов уравнений
            if (jsonData.coefs) {
                jsonData.coefs.forEach((coef, index) => {
                    document.getElementById(`a${index + 1}`).value = coef[0];
                    document.getElementById(`b${index + 1}`).value = coef[1];
                    document.getElementById(`c${index + 1}`).value = coef[2];
                    document.getElementById(`d${index + 1}`).value = coef[3];
                });
            }

            alert('Данные успешно загружены из файла!');
        } catch (error) {
            alert('Ошибка при загрузке файла: ' + error.message);
        }

        // Удаляем временный input
        event.target.remove();
    };

    reader.readAsText(file);
}

// Функция вычисления результата
async function calculate() {
    const calculateBtn = document.getElementById("calculate");
    const originalText = calculateBtn.textContent;
    
    try {
        calculateBtn.textContent = "Вычисление...";
        calculateBtn.disabled = true;

        const formData = new FormData();

        // Получаем количество лет для расчёта
        const years = parseInt(document.getElementById('yearsInput').value) || 10;
        formData.append('years', years);

        // Собираем значения X1-X8 (startValues)
        const startValues = [];
        for (let i = 1; i <= NUM_VARIABLES; i++) {
            const value = parseFloat(document.getElementById(`X${i}`).value) || 0;
            startValues.push(value);
        }
        formData.append('startValues', JSON.stringify(startValues));

        // Собираем значения X1-X8 MAX (maxValues)
        const maxValues = [];
        for (let i = 1; i <= NUM_VARIABLES; i++) {
            const value = parseFloat(document.getElementById(`X${i}_max`).value) || 0;
            maxValues.push(value);
        }
        formData.append('maxValues', JSON.stringify(maxValues));

        // Собираем значения X1-X8 norm (normValues)
        const normValues = [];
        for (let i = 1; i <= NUM_VARIABLES; i++) {
            const value = parseFloat(document.getElementById(`X${i}_norm`).value) || 1.0;
            normValues.push(value);
        }
        formData.append('normValues', JSON.stringify(normValues));

        // Собираем коэффициенты возмущений
        const qcoefs = [];
        for (let i = 1; i <= NUM_DISTURBANCES; i++) {
            const qa = parseFloat(document.getElementById(`qa${i}`).value) || 0;
            const qb = parseFloat(document.getElementById(`qb${i}`).value) || 0;
            const qc = parseFloat(document.getElementById(`qc${i}`).value) || 0;
            const qd = parseFloat(document.getElementById(`qd${i}`).value) || 0;
            qcoefs.push([qa, qb, qc, qd]);
        }
        formData.append('qcoefs', JSON.stringify(qcoefs));

        // Собираем коэффициенты уравнений
        const coefs = [];
        for (let i = 1; i <= NUM_EQUATIONS; i++) {
            const a = parseFloat(document.getElementById(`a${i}`).value) || 0;
            const b = parseFloat(document.getElementById(`b${i}`).value) || 0;
            const c = parseFloat(document.getElementById(`c${i}`).value) || 0;
            const d = parseFloat(document.getElementById(`d${i}`).value) || 0;
            coefs.push([a, b, c, d]);
        }
        formData.append('coefs', JSON.stringify(coefs));
        
        // Отправляем запрос
        const response = await fetch("/calculate/", {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Отображаем результаты
        const resultDiv = document.getElementById("results");

        // Форматируем новый результат с машинными графиками
        const newResult = `
            <div class="result-item">
                <h3 class="text-center mt-3 mb-3">Линейный график</h3>
                <div class="row">
                    <div class="col-md-12">
                        <img src="data:image/png;base64,${data.image1}" alt="Машинные координаты" class="img-fluid zoomable-image js-zoomable">
                        <p class="text-center"><small>График: Зависимость машинных координат от времени (ось Y: 0-1, ось X: 0-конец расчётов)</small></p>
                    </div>
                </div>
                <h3 class="text-center mt-4 mb-3">Радиальные диаграммы для шагов координат</h3>
                <div class="row">
                    <div class="col-md-12">
                        <img src="data:image/png;base64,${data.image2}" alt="Радиальные диаграммы" class="img-fluid zoomable-image js-zoomable">
                        <p class="text-center"><small>6 радиальных диаграмм для координат: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0</small></p>
                    </div>
                </div>
            </div>
        `;

        // Очищаем старые результаты и добавляем новый
        resultDiv.innerHTML = newResult;

        alert('Расчет выполнен успешно!');
        
    } catch (error) {
        console.error('Ошибка:', error);
        alert('Ошибка при выполнении расчета: ' + error.message);
    } finally {
        calculateBtn.textContent = originalText;
        calculateBtn.disabled = false;
    }
}

// Функция очистки предыдущих результатов
function clearPreviousResults() {
    const resultDiv = document.getElementById("results");
    resultDiv.innerHTML = '<p class="text-muted">Выполните расчет для отображения графиков машинного времени и координат.</p>';
}

function setupImageZoom() {
    const resultsContainer = document.getElementById("results");
    const modal = document.getElementById("imageModal");
    const modalImg = document.getElementById("imageModalImg");
    const closeBtn = document.getElementById("imageModalClose");

    if (!resultsContainer || !modal || !modalImg || !closeBtn) {
        return;
    }

    const closeModal = () => {
        modal.classList.remove("open");
        modal.setAttribute("aria-hidden", "true");
        document.body.classList.remove("modal-open");
    };

    const openModal = (src, alt) => {
        modalImg.src = src;
        modalImg.alt = alt || "Увеличенное изображение";
        modal.classList.add("open");
        modal.setAttribute("aria-hidden", "false");
        document.body.classList.add("modal-open");
    };

    resultsContainer.addEventListener("click", (event) => {
        const target = event.target;
        if (target && target.classList.contains("js-zoomable")) {
            openModal(target.src, target.alt);
        }
    });

    modal.addEventListener("click", (event) => {
        if (event.target === modal) {
            closeModal();
        }
    });

    closeBtn.addEventListener("click", closeModal);

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && modal.classList.contains("open")) {
            closeModal();
        }
    });
}
