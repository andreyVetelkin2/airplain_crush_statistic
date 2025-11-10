"""
Скрипт для тестирования интеграции фронтенда и бэкенда
"""

import requests
import json
import time

# Ждем пока сервер запустится
print("Ожидание запуска сервера...")
time.sleep(3)

BASE_URL = "http://localhost:8000"

def test_health():
    """Тест проверки работоспособности"""
    print("\n=== Тест 1: Проверка работоспособности API ===")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            print("✓ API работает нормально")
            print(f"  Ответ: {response.json()}")
            return True
        else:
            print(f"✗ Ошибка: статус {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Ошибка подключения: {e}")
        return False

def test_calculate_endpoint():
    """Тест endpoint расчета"""
    print("\n=== Тест 2: Проверка endpoint /calculate/ ===")
    
    # Подготавливаем тестовые данные
    start_values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    max_values = [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25]
    norm_values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    # Коэффициенты возмущений (5 возмущений)
    qcoefs = []
    for i in range(5):
        qcoefs.append([0.01, 0.02, 0.03, 1.0])
    
    # Коэффициенты уравнений (20 уравнений)
    coefs = []
    for i in range(20):
        coefs.append([0.01, 0.02, 0.03, 0.5])
    
    # Формируем данные для отправки
    form_data = {
        'startValues': json.dumps(start_values),
        'maxValues': json.dumps(max_values),
        'normValues': json.dumps(norm_values),
        'qcoefs': json.dumps(qcoefs),
        'coefs': json.dumps(coefs)
    }
    
    try:
        response = requests.post(f"{BASE_URL}/calculate/", data=form_data)
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Расчет выполнен успешно")
            print(f"  Статус: {data.get('status')}")
            print(f"  График 1: {len(data.get('image1', ''))} символов (base64)")
            print(f"  График 2: {len(data.get('image2', ''))} символов (base64)")
            
            # Проверяем что графики не пустые
            if len(data.get('image1', '')) > 100 and len(data.get('image2', '')) > 100:
                print("✓ Графики успешно сгенерированы")
                return True
            else:
                print("✗ Графики пустые или слишком маленькие")
                return False
        else:
            print(f"✗ Ошибка: статус {response.status_code}")
            print(f"  Ответ: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Ошибка при выполнении запроса: {e}")
        return False

def test_static_files():
    """Тест доступности статических файлов"""
    print("\n=== Тест 3: Проверка статических файлов ===")
    
    files = [
        '/static/styles.css',
        '/static/app.js'
    ]
    
    all_ok = True
    for file_path in files:
        try:
            response = requests.get(f"{BASE_URL}{file_path}")
            if response.status_code == 200:
                print(f"✓ {file_path} доступен ({len(response.text)} символов)")
            else:
                print(f"✗ {file_path} недоступен (статус {response.status_code})")
                all_ok = False
        except Exception as e:
            print(f"✗ Ошибка при запросе {file_path}: {e}")
            all_ok = False
    
    return all_ok

def test_main_page():
    """Тест главной страницы"""
    print("\n=== Тест 4: Проверка главной страницы ===")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            html = response.text
            
            # Проверяем наличие ключевых элементов
            checks = [
                ('Bootstrap', 'bootstrap' in html.lower()),
                ('Таблица показателей', 'исследуемые показатели' in html.lower()),
                ('Таблица возмущений', 'возмущения' in html.lower()),
                ('Таблица уравнений', 'уравнения связей' in html.lower()),
                ('Кнопка расчета', 'вычислить результат' in html.lower()),
                ('Случайные значения', 'случайными значениями' in html.lower())
            ]
            
            all_ok = True
            for check_name, check_result in checks:
                if check_result:
                    print(f"  ✓ {check_name}")
                else:
                    print(f"  ✗ {check_name} не найден")
                    all_ok = False
            
            return all_ok
        else:
            print(f"✗ Ошибка: статус {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Ошибка при запросе главной страницы: {e}")
        return False

def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ СИСТЕМЫ")
    print("=" * 60)
    
    results = []
    
    # Запускаем тесты
    results.append(("Проверка API", test_health()))
    results.append(("Главная страница", test_main_page()))
    results.append(("Статические файлы", test_static_files()))
    results.append(("Endpoint расчета", test_calculate_endpoint()))
    
    # Подводим итоги
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nРезультат: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("\nСистема готова к использованию:")
        print(f"  • Откройте браузер и перейдите по адресу: {BASE_URL}")
        print("  • Нажмите 'Заполнить случайными значениями' для генерации данных")
        print("  • Нажмите 'Вычислить результат' для запуска расчета")
        print("  • Используйте 'Сохранить' и 'Загрузить' для работы с данными")
    else:
        print("\n⚠️  Некоторые тесты не прошли. Проверьте ошибки выше.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nТестирование прервано пользователем")
        exit(1)

