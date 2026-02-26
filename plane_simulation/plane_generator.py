import json
import os
import sys

# Настройка путей для корректного импорта при любом способе запуска
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Теперь импортируем как пакет
try:
    from plane_simulation.plane_simulation import simulate
except ImportError:
    # На случай если мы всё же не можем найти пакет, пробуем прямой импорт
    from plane_simulation import simulate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(BASE_DIR, "simulation_result.json")

def generate_planes(count=5):
    """
    Генерирует заданное количество самолетов и сохраняет их в JSON.
    """
    all_results = []
    
    print(f"Генерация {count} самолетов...")
    
    for i in range(1, count + 1):
        print(f"\n--- Генерация самолета №{i} ---")
        try:
            # Вызываем симуляцию для каждого самолета
            result, _ = simulate(plane_number=i)
            all_results.append(result)
        except Exception as e:
            print(f"Ошибка при генерации самолета №{i}: {e}")
    
    # Сохраняем все результаты в один файл
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nВсего сгенерировано {len(all_results)} самолетов.")
    print(f"Результаты сохранены в {RESULT_FILE}")
    return all_results

if __name__ == "__main__":
    generate_planes()
