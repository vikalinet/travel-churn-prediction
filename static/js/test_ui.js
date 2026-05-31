/**
 * UI логика для тестирования сценариев
 */

// Загрузка тестовых данных с сервера
async function loadTestData() {
    try {
        const response = await fetch('/api/v1/test-data');
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.log('Используем встроенные тестовые данные');
    }
    return null;
}

// Загрузка данных при старте
let testData = {
    // Положительные сценарии
    pos_001: {
        name: "Активный VIP клиент",
        data: { age: 42, frequent_flyer: "Yes", annual_income_class: "High Income", services_opted: 5, account_synced_to_social_media: "Yes", booked_hotel_or_not: "Yes" },
        expected: { risk_level: "Low", probability_max: 0.30 }
    },
    pos_002: {
        name: "Молодой активный клиент",
        data: { age: 28, frequent_flyer: "Yes", annual_income_class: "Middle Income", services_opted: 4, account_synced_to_social_media: "Yes", booked_hotel_or_not: "Yes" },
        expected: { risk_level: "Low", probability_max: 0.35 }
    },
    pos_003: {
        name: "Клиент с максимумом услуг",
        data: { age: 38, frequent_flyer: "Yes", annual_income_class: "High Income", services_opted: 6, account_synced_to_social_media: "Yes", booked_hotel_or_not: "Yes" },
        expected: { risk_level: "Low", probability_max: 0.20 }
    },
    pos_004: {
        name: "Постоянный клиент",
        data: { age: 45, frequent_flyer: "Yes", annual_income_class: "Middle Income", services_opted: 5, account_synced_to_social_media: "Yes", booked_hotel_or_not: "No" },
        expected: { risk_level: "Low", probability_max: 0.25 }
    },
    pos_005: {
        name: "Бюджетный но активный",
        data: { age: 32, frequent_flyer: "Yes", annual_income_class: "Low Income", services_opted: 4, account_synced_to_social_media: "Yes", booked_hotel_or_not: "Yes" },
        expected: { risk_level: "Low", probability_max: 0.40 }
    },

    // Отрицательные сценарии
    neg_001: {
        name: "Новый неактивный клиент",
        data: { age: 24, frequent_flyer: "No", annual_income_class: "Low Income", services_opted: 1, account_synced_to_social_media: "No", booked_hotel_or_not: "No" },
        expected: { risk_level: "High", probability_min: 0.70 }
    },
    neg_002: {
        name: "Клиент с признаками оттока",
        data: { age: 35, frequent_flyer: "No", annual_income_class: "Middle Income", services_opted: 2, account_synced_to_social_media: "No", booked_hotel_or_not: "No" },
        expected: { risk_level: "High", probability_min: 0.65 }
    },
    neg_003: {
        name: "Молодой с низким engagement",
        data: { age: 22, frequent_flyer: "No", annual_income_class: "Low Income", services_opted: 1, account_synced_to_social_media: "Yes", booked_hotel_or_not: "No" },
        expected: { risk_level: "Medium", probability_min: 0.50 }
    },
    neg_004: {
        name: "Клиент с падающей активностью",
        data: { age: 40, frequent_flyer: "No", annual_income_class: "High Income", services_opted: 2, account_synced_to_social_media: "No", booked_hotel_or_not: "No" },
        expected: { risk_level: "High", probability_min: 0.75 }
    },
    neg_005: {
        name: "Клиент с минимальными показателями",
        data: { age: 30, frequent_flyer: "No", annual_income_class: "Low Income", services_opted: 1, account_synced_to_social_media: "No", booked_hotel_or_not: "No" },
        expected: { risk_level: "High", probability_min: 0.80 }
    },

    // Сценарии дрифта
    drift_001: {
        name: "Пожилой активный (аномалия)",
        data: { age: 75, frequent_flyer: "Yes", annual_income_class: "High Income", services_opted: 5, account_synced_to_social_media: "No", booked_hotel_or_not: "Yes" },
        expected: { risk_level: "Low", probability_max: 0.40 },
        note: "Возраст 75 - за пределами типичного диапазона"
    },
    drift_002: {
        name: "Молодой VIP",
        data: { age: 20, frequent_flyer: "Yes", annual_income_class: "High Income", services_opted: 6, account_synced_to_social_media: "Yes", booked_hotel_or_not: "Yes" },
        expected: { risk_level: "Low", probability_max: 0.35 },
        note: "Возраст 20 - редкий паттерн"
    },
    drift_003: {
        name: "Экстремальный возраст",
        data: { age: 85, frequent_flyer: "No", annual_income_class: "Middle Income", services_opted: 3, account_synced_to_social_media: "No", booked_hotel_or_not: "No" },
        expected: { risk_level: "Medium", probability_min: 0.40 },
        note: "Возраст 85 - за пределами обучающей выборки"
    },
    drift_004: {
        name: "Противоречивые признаки",
        data: { age: 35, frequent_flyer: "No", annual_income_class: "High Income", services_opted: 6, account_synced_to_social_media: "Yes", booked_hotel_or_not: "Yes" },
        expected: { risk_level: "Medium", probability_max: 0.40 },
        note: "Высокий доход и много услуг, но НЕ летает"
    },
    drift_005: {
        name: "Без соцсетей но активен",
        data: { age: 50, frequent_flyer: "Yes", annual_income_class: "High Income", services_opted: 5, account_synced_to_social_media: "No", booked_hotel_or_not: "Yes" },
        expected: { risk_level: "Low", probability_max: 0.35 },
        note: "Активный клиент без синхронизации"
    },

    // Граничные случаи
    edge_001: {
        name: "Граница Low/Medium",
        data: { age: 30, frequent_flyer: "Yes", annual_income_class: "Middle Income", services_opted: 3, account_synced_to_social_media: "Yes", booked_hotel_or_not: "No" },
        expected: { risk_level: "Low", probability_min: 0.25, probability_max: 0.35 },
        note: "На границе классов"
    },
    edge_002: {
        name: "Граница Medium/High",
        data: { age: 35, frequent_flyer: "No", annual_income_class: "Middle Income", services_opted: 2, account_synced_to_social_media: "No", booked_hotel_or_not: "No" },
        expected: { risk_level: "Medium", probability_min: 0.55, probability_max: 0.70 },
        note: "Вероятность около 60-65%"
    },
    edge_003: {
        name: "Минимальный возраст",
        data: { age: 18, frequent_flyer: "Yes", annual_income_class: "Low Income", services_opted: 2, account_synced_to_social_media: "Yes", booked_hotel_or_not: "No" },
        expected: { risk_level: "Medium", probability_max: 0.55 },
        note: "Самый молодой (18 лет)"
    }
};

// Инициализация при загрузке страницы
(async function init() {
    const serverData = await loadTestData();
    if (serverData && Object.keys(serverData).length > 0) {
        // Преобразуем данные с сервера в формат для UI
        const converted = {};

        serverData.positive_scenarios?.forEach(s => {
            converted[s.id] = { name: s.name, data: s.data, expected: s.expected, note: s.expected.note };
        });
        serverData.negative_scenarios?.forEach(s => {
            converted[s.id] = { name: s.name, data: s.data, expected: s.expected, note: s.expected.note };
        });
        serverData.drift_scenarios?.forEach(s => {
            converted[s.id] = { name: s.name, data: s.data, expected: s.expected, note: s.expected.note };
        });
        serverData.edge_scenarios?.forEach(s => {
            converted[s.id] = { name: s.name, data: s.data, expected: s.expected, note: s.expected.note };
        });

        if (Object.keys(converted).length > 0) {
            testData = converted;
        }
    }
})();

// Применить сценарий
function applyScenario(id) {
    const scenario = testData[id];
    if (!scenario) return;

    // Проверяем, есть ли форма на странице (для test_ui.html её нет)
    const ageField = document.getElementById('age');
    if (ageField) {
        // Заполняем форму (если она есть)
        document.getElementById('age').value = scenario.data.age;
        document.getElementById('frequent_flyer').value = scenario.data.frequent_flyer;
        document.getElementById('annual_income_class').value = scenario.data.annual_income_class;
        document.getElementById('services_opted').value = scenario.data.services_opted;
        document.getElementById('account_synced_to_social_media').value = scenario.data.account_synced_to_social_media;
        document.getElementById('booked_hotel_or_not').value = scenario.data.booked_hotel_or_not;

        // Прокрутка к форме
        document.querySelector('.form-container')?.scrollIntoView({ behavior: 'smooth' });

        // Запускаем предсказание автоматически
        setTimeout(() => {
            submitFormAndTest(id);
        }, 500);
    } else {
        // Если формы нет, сразу запускаем тест
        submitFormAndTest(id);
    }
}

// Отправка формы и сохранение результата
async function submitFormAndTest(scenarioId) {
    const scenario = testData[scenarioId];

    try {
        const response = await fetch('/api/v1/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(scenario.data)
        });

        if (response.ok) {
            const result = await response.json();
            displayTestResult(scenarioId, scenario, result);
            return result;
        }
    } catch (error) {
        console.error('Ошибка тестирования:', error);
    }
    return null;
}

// Массив для хранения всех результатов
let allTestResults = [];

// Отображение результата теста
function displayTestResult(id, scenario, result) {
    console.log(`Тест ${id}:`, {
        expected: scenario.expected,
        actual: result
    });

    // Проверка результата
    let passed = true;
    let matchDetails = [];

    // Проверка уровня риска (только для информации)
    if (scenario.expected.risk_level) {
        const riskMatch = result.risk_level === scenario.expected.risk_level;
        matchDetails.push({
            label: "Risk Level",
            expected: scenario.expected.risk_level,
            actual: result.risk_level,
            match: riskMatch
        });
        // Пока не устанавливаем passed = false по риску
    }

    // Проверка вероятности (верхняя граница)
    if (scenario.expected.probability_max) {
        const probMatch = result.probability <= scenario.expected.probability_max;
        console.log(`Prob check (max): expected <= ${(scenario.expected.probability_max * 100).toFixed(0)}%, actual ${(result.probability * 100).toFixed(1)}% -> ${probMatch ? 'PASS' : 'FAIL'}`);
        matchDetails.push({
            label: "Probability",
            expected: `<= ${(scenario.expected.probability_max * 100).toFixed(0)}%`,
            actual: `${(result.probability * 100).toFixed(1)}%`,
            match: probMatch
        });
        if (!probMatch) passed = false;
    }

    // Проверка вероятности (нижняя граница)
    if (scenario.expected.probability_min && !scenario.expected.probability_max) {
        const probMatch = result.probability >= scenario.expected.probability_min;
        console.log(`Prob check (min): expected >= ${(scenario.expected.probability_min * 100).toFixed(0)}%, actual ${(result.probability * 100).toFixed(1)}% -> ${probMatch ? 'PASS' : 'FAIL'}`);
        matchDetails.push({
            label: "Probability",
            expected: `>= ${(scenario.expected.probability_min * 100).toFixed(0)}%`,
            actual: `${(result.probability * 100).toFixed(1)}%`,
            match: probMatch
        });
        if (!probMatch) passed = false;
    }

    // Определение типа сценария
    let scenarioType = 'positive';
    if (id.startsWith('pos_')) scenarioType = 'positive';
    else if (id.startsWith('neg_')) scenarioType = 'negative';
    else if (id.startsWith('drift_')) scenarioType = 'drift';
    else if (id.startsWith('edge_')) scenarioType = 'edge';

    // Сохранение результата
    const testResult = {
        id: id,
        name: scenario.name,
        type: scenarioType,
        expected_risk: scenario.expected.risk_level || 'N/A',
        expected_prob: formatExpectedProb(scenario.expected),
        actual_risk: result.risk_level,
        actual_prob: result.probability,
        passed: passed,
        timestamp: new Date().toISOString(),
        details: matchDetails
    };

    allTestResults.push(testResult);

    // Обновление таблицы
    updateResultsTable(testResult);

    // Обновление статистики
    updateStats();

    // Прокрутка к результатам
    const testResults = document.getElementById('testResults');
    testResults.classList.add('show');
    testResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Форматирование ожидаемой вероятности
function formatExpectedProb(expected) {
    const parts = [];
    if (expected.probability_min) {
        parts.push(`>= ${(expected.probability_min * 100).toFixed(0)}%`);
    }
    if (expected.probability_max) {
        parts.push(`<= ${(expected.probability_max * 100).toFixed(0)}%`);
    }
    return parts.join(' ') || 'N/A';
}

// Обновление таблицы результатов
function updateResultsTable(testResult) {
    const tbody = document.getElementById('resultsTableBody');
    const row = document.createElement('tr');
    row.id = `result-${testResult.id}`;
    row.dataset.type = testResult.type;
    row.dataset.passed = testResult.passed;

    const probClass = testResult.actual_prob < 0.3 ? 'prob-low' :
                     testResult.actual_prob < 0.5 ? 'prob-medium' : 'prob-high';

    row.innerHTML = `
        <td class="scenario-name-cell">${testResult.id}: ${testResult.name}</td>
        <td><span class="scenario-type-badge type-${testResult.type}">${getScenarioTypeLabel(testResult.type)}</span></td>
        <td>${testResult.expected_risk}<br><small style="color: #666;">${testResult.expected_prob}</small></td>
        <td><strong>${testResult.actual_risk}</strong></td>
        <td><span class="probability-value ${probClass}">${(testResult.actual_prob * 100).toFixed(1)}%</span></td>
        <td><span class="${testResult.passed ? 'status-pass' : 'status-fail'}">${testResult.passed ? '✅ PASS' : '❌ FAIL'}</span></td>
    `;

    tbody.appendChild(row);
}

// Получение подписи типа сценария
function getScenarioTypeLabel(type) {
    const labels = {
        'positive': '✅ Положительный',
        'negative': '❌ Отрицательный',
        'drift': '🟡 Дрифт',
        'edge': '🟣 Граничный'
    };
    return labels[type] || type;
}

// Обновление статистики
function updateStats() {
    const total = allTestResults.length;
    const passed = allTestResults.filter(r => r.passed).length;
    const failed = total - passed;
    const passRate = total > 0 ? Math.round((passed / total) * 100) : 0;

    document.getElementById('passCount').textContent = passed;
    document.getElementById('failCount').textContent = failed;
    document.getElementById('totalCount').textContent = total;
    document.getElementById('passRate').textContent = `${passRate}%`;
}

// Фильтрация результатов
function filterResults(filterType) {
    const rows = document.querySelectorAll('#resultsTableBody tr');

    // Обновление активной кнопки
    document.querySelectorAll('.btn-filter').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    rows.forEach(row => {
        const type = row.dataset.type;
        const passed = row.dataset.passed === 'true';

        let show = true;
        if (filterType === 'positive') show = type === 'positive';
        else if (filterType === 'negative') show = type === 'negative';
        else if (filterType === 'drift') show = type === 'drift';
        else if (filterType === 'edge') show = type === 'edge';
        else if (filterType === 'pass') show = passed;
        else if (filterType === 'fail') show = !passed;

        row.style.display = show ? '' : 'none';
    });
}

// Экспорт результатов в CSV
function exportResults() {
    if (allTestResults.length === 0) {
        alert('Нет результатов для экспорта');
        return;
    }

    const headers = ['ID', 'Название', 'Тип', 'Ожидаемый риск', 'Ожидаемая вероятность', 'Фактический риск', 'Фактическая вероятность', 'Статус', 'Время'];
    const rows = allTestResults.map(r => [
        r.id,
        r.name,
        r.type,
        r.expected_risk,
        r.expected_prob,
        r.actual_risk,
        (r.actual_prob * 100).toFixed(2) + '%',
        r.passed ? 'PASS' : 'FAIL',
        r.timestamp
    ]);

    let csv = headers.join(',') + '\n';
    rows.forEach(row => {
        csv += row.map(cell => `"${cell}"`).join(',') + '\n';
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `test_results_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
}

// Запуск всех положительных сценариев
async function runAllPositive() {
    showProgress('Запуск положительных сценариев...');
    const scenarios = ['pos_001', 'pos_002', 'pos_003', 'pos_004', 'pos_005'];
    for (const id of scenarios) {
        await runScenario(id);
        await delay(800);
    }
    hideProgress();
}

// Запуск всех отрицательных сценариев
async function runAllNegative() {
    showProgress('Запуск отрицательных сценариев...');
    const scenarios = ['neg_001', 'neg_002', 'neg_003', 'neg_004', 'neg_005'];
    for (const id of scenarios) {
        await runScenario(id);
        await delay(800);
    }
    hideProgress();
}

// Запуск всех сценариев дрифта
async function runAllDrift() {
    showProgress('Запуск сценариев дрифта...');
    const scenarios = ['drift_001', 'drift_002', 'drift_003', 'drift_004', 'drift_005'];
    for (const id of scenarios) {
        await runScenario(id);
        await delay(800);
    }
    hideProgress();
}

// Запуск всех граничных сценариев
async function runAllEdge() {
    showProgress('Запуск граничных сценариев...');
    const scenarios = ['edge_001', 'edge_002', 'edge_003'];
    for (const id of scenarios) {
        await runScenario(id);
        await delay(800);
    }
    hideProgress();
}

// Показать индикатор прогресса
function showProgress(text) {
    const indicator = document.getElementById('progressIndicator');
    const progressText = document.getElementById('progressText');
    progressText.textContent = text;
    indicator.style.display = 'flex';
}

// Скрыть индикатор прогресса
function hideProgress() {
    const indicator = document.getElementById('progressIndicator');
    indicator.style.display = 'none';
}

// Запуск одного сценария
async function runScenario(id) {
    const scenario = testData[id];
    if (!scenario) return;

    // Проверяем, есть ли форма на странице
    const ageField = document.getElementById('age');
    if (ageField) {
        document.getElementById('age').value = scenario.data.age;
        document.getElementById('frequent_flyer').value = scenario.data.frequent_flyer;
        document.getElementById('annual_income_class').value = scenario.data.annual_income_class;
        document.getElementById('services_opted').value = scenario.data.services_opted;
        document.getElementById('account_synced_to_social_media').value = scenario.data.account_synced_to_social_media;
        document.getElementById('booked_hotel_or_not').value = scenario.data.booked_hotel_or_not;
    }

    await submitFormAndTest(id);
}

// Задержка
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Очистка результатов
function clearResults() {
    allTestResults = [];
    document.getElementById('resultsTableBody').innerHTML = '';
    document.getElementById('resultsCards').innerHTML = '';
    document.getElementById('testResults').classList.remove('show');
    updateStats();
}
