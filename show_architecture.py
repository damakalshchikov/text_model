#!/usr/bin/env python3
"""
Демонстрация архитектуры проекта "Предсказатель цен на дома"
"""

from pathlib import Path


def show_tree(directory, prefix="", max_depth=3, current_depth=0):
    """Показать дерево файлов проекта"""
    if current_depth > max_depth:
        return

    path = Path(directory)
    if not path.exists():
        return

    items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))

    for i, item in enumerate(items):
        is_last = i == len(items) - 1

        if item.name.startswith(".") and item.name not in [".gitignore"]:
            continue

        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}")

        if item.is_dir() and current_depth < max_depth:
            extension = "    " if is_last else "│   "
            show_tree(item, prefix + extension, max_depth, current_depth + 1)


def show_architecture_summary():
    """Показать сводку по архитектуре"""
    print("🏗️  АРХИТЕКТУРА ПРОЕКТА 'Предсказатель цен на дома'")
    print("=" * 60)

    print("\n📁 Структура файлов:")
    show_tree(".", max_depth=3)

    print("\n🏛️  Принципы архитектуры:")
    print("┌─ ✅ МОДУЛЬНОСТЬ - четкое разделение ответственности")
    print("├─ ✅ КЭШИРОВАНИЕ - оптимизация производительности")
    print("├─ ✅ ВАЛИДАЦИЯ - проверка входных данных")
    print("├─ ✅ ОБРАБОТКА ОШИБОК - graceful error handling")
    print("├─ ✅ КОНФИГУРАЦИЯ - централизованные настройки")
    print("└─ ✅ ДОКУМЕНТАЦИЯ - подробное описание")

    print("\n🚀 Готовые к запуску версии:")
    print("┌─ app_mvp.py     → MVP версия (рекомендуется)")
    print("├─ app.py         → Оригинальная версия")
    print("└─ main.ipynb     → Jupyter ноутбук для анализа")

    print("\n🔄 Модульная архитектура (в разработке):")
    print("┌─ src/config/    → Конфигурационные файлы")
    print("├─ src/data/      → Управление данными")
    print("├─ src/models/    → ML модели")
    print("├─ src/utils/     → Утилиты и форматирование")
    print("└─ src/main.py    → Главный файл модульной версии")

    print("\n📊 Технические характеристики:")
    print("┌─ 🧠 Модель: Random Forest Regressor")
    print("├─ 📈 R² ≈ 0.67 (67% объясненной дисперсии)")
    print("├─ 🎯 Данные: California Housing Dataset (~20k образцов)")
    print("├─ ⚡ Производительность: кэширование + параллелизм")
    print("└─ 🔒 Надежность: валидация + обработка ошибок")

    print("\n🎨 Интерфейс:")
    print("┌─ 🔮 Вкладка предсказания с интерактивными слайдерами")
    print("├─ 📊 Аналитика с графиками важности признаков")
    print("├─ 📈 Визуализация точности модели")
    print("└─ ⚙️ Техническая информация об архитектуре")

    print("\n" + "=" * 60)
    print("🚀 Для запуска: streamlit run app_mvp.py")
    print("📚 Подробности: см. README.md")


if __name__ == "__main__":
    show_architecture_summary()
