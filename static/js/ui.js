/**
 * UI JavaScript for Travel Churn Prediction Interface
 * Логика интерактивного веб-интерфейса
 */

/**
 * Отправка формы предсказания
 */
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');

    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
});

/**
 * Обработчик отправки формы
 * @param {Event} e - Событие отправки формы
 */
async function handleFormSubmit(e) {
    e.preventDefault();

    const submitBtn = document.getElementById('submitBtn');
    const loading = document.getElementById('loading');
    const errorMessage = document.getElementById('errorMessage');
    const resultContainer = document.getElementById('resultContainer');

    // Сброс предыдущих состояний
    submitBtn.disabled = true;
    loading.classList.add('show');
    errorMessage.classList.remove('show');
    resultContainer.classList.remove('show');

    // Сбор данных из формы
    const formData = collectFormData();

    try {
        const response = await fetch('/api/v1/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Ошибка при получении прогноза');
        }

        const result = await response.json();
        displayResult(result, formData);

    } catch (error) {
        showError(error.message);
    } finally {
        submitBtn.disabled = false;
        loading.classList.remove('show');
    }
}

/**
 * Сбор данных из формы
 * @returns {Object} Данные клиента
 */
function collectFormData() {
    return {
        age: parseInt(document.getElementById('age').value),
        frequent_flyer: document.getElementById('frequent_flyer').value,
        annual_income_class: document.getElementById('annual_income_class').value,
        services_opted: parseInt(document.getElementById('services_opted').value),
        account_synced_to_social_media: document.getElementById('account_synced_to_social_media').value,
        booked_hotel_or_not: document.getElementById('booked_hotel_or_not').value
    };
}

/**
 * Отображение результатов предсказания
 * @param {Object} result - Результат от API
 * @param {Object} formData - Данные формы
 */
function displayResult(result, formData) {
    // Основные элементы
    const resultContainer = document.getElementById('resultContainer');
    const riskLevel = document.getElementById('riskLevel');
    const probabilityValue = document.getElementById('probabilityValue');
    const probabilityBar = document.getElementById('probabilityBar');
    const riskDescription = document.getElementById('riskDescription');
    const recommendationText = document.getElementById('recommendationText');

    // Установка уровня риска
    const riskEmoji = result.risk_level === 'Low' ? '🟢 НИЗКИЙ' :
                      result.risk_level === 'Medium' ? '🟡 СРЕДНИЙ' : '🔴 ВЫСОКИЙ';

    riskLevel.textContent = riskEmoji;
    riskLevel.className = 'prediction-value prediction-' + result.risk_level.toLowerCase();

    // Установка вероятности
    const probabilityPercent = (result.probability * 100).toFixed(1);
    probabilityValue.textContent = probabilityPercent + '%';
    probabilityBar.style.width = probabilityPercent + '%';

    // Описание и рекомендации
    setRecommendation(result.risk_level, riskDescription, recommendationText);

    // Отображение метрик модели
    displayMetrics(result.metrics);

    // Отображение деталей клиента
    displayCustomerDetails(formData);

    // Показ результатов с анимацией
    resultContainer.classList.add('show');

    // Прокрутка к результатам
    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Установка рекомендаций в зависимости от уровня риска
 * @param {string} riskLevel - Уровень риска (Low/Medium/High)
 * @param {HTMLElement} descriptionEl - Элемент описания
 * @param {HTMLElement} recommendationEl - Элемент рекомендации
 */
function setRecommendation(riskLevel, descriptionEl, recommendationEl) {
    if (riskLevel === 'Low') {
        descriptionEl.textContent = 'Клиент с высокой вероятностью останется с вами';
        recommendationEl.textContent = '✅ Клиент лоялен. Предложите программу лояльности для укрепления отношений и увеличения частоты покупок.';
    } else if (riskLevel === 'Medium') {
        descriptionEl.textContent = 'Клиент может уйти, но есть шанс его удержать';
        recommendationEl.textContent = '⚠️ Рекомендуется персональное предложение со скидкой 10-15% и звонок от менеджера для выяснения потребностей.';
    } else {
        descriptionEl.textContent = 'Высокий риск потери клиента! Требуется срочное вмешательство';
        recommendationEl.textContent = '🚨 Немедленно предложите специальную акцию со скидкой 20-30%, персонального менеджера и эксклюзивные предложения.';
    }
}

/**
 * Отображение метрик модели
 * @param {Object} metrics - Метрики модели
 */
function displayMetrics(metrics) {
    if (metrics && Object.keys(metrics).length > 0) {
        document.getElementById('metricAccuracy').textContent = formatPercent(metrics.accuracy);
        document.getElementById('metricPrecision').textContent = formatPercent(metrics.precision);
        document.getElementById('metricRecall').textContent = formatPercent(metrics.recall);
        document.getElementById('metricF1').textContent = formatPercent(metrics.f1_score);
        document.getElementById('metricROC').textContent = formatPercent(metrics.roc_auc);
    }
}

/**
 * Форматирование процентов
 * @param {number} value - Значение от 0 до 1
 * @returns {string} Отформатированный процент
 */
function formatPercent(value) {
    if (value === undefined || value === null) return '--';
    return (value * 100).toFixed(1) + '%';
}

/**
 * Отображение деталей клиента
 * @param {Object} formData - Данные клиента
 */
function displayCustomerDetails(formData) {
    document.getElementById('detailAge').textContent = formData.age + ' лет';
    document.getElementById('detailFlyer').textContent = formData.frequent_flyer === 'Yes' ? 'Да ✈️' : 'Нет';
    document.getElementById('detailIncome').textContent = formData.annual_income_class;
    document.getElementById('detailServices').textContent = formData.services_opted;
    document.getElementById('detailSync').textContent = formData.account_synced_to_social_media === 'Yes' ? 'Да 📱' : 'Нет';
    document.getElementById('detailHotel').textContent = formData.booked_hotel_or_not === 'Yes' ? 'Да 🏨' : 'Нет';
}

/**
 * Показ сообщения об ошибке
 * @param {string} message - Текст ошибки
 */
function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = '❌ ' + message;
    errorMessage.classList.add('show');
}

/**
 * Сброс формы и результатов
 */
function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('resultContainer').classList.remove('show');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Экспорт функции для глобального доступа из HTML
window.resetForm = resetForm;
