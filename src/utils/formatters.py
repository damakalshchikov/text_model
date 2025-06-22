"""Утилиты для форматирования и валидации данных"""

from typing import Any, Dict, Optional


def format_currency(amount: float, currency: str = "$") -> str:
    """
    Форматирует число как валюту

    Args:
        amount: Сумма для форматирования
        currency: Символ валюты

    Returns:
        Отформатированная строка
    """
    return f"{currency}{amount:,.0f}"


def format_number(number: float, decimal_places: int = 2) -> str:
    """
    Форматирует число с разделителем тысяч

    Args:
        number: Число для форматирования
        decimal_places: Количество знаков после запятой

    Returns:
        Отформатированная строка
    """
    format_str = f"{{:,.{decimal_places}f}}"
    return format_str.format(number)


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Форматирует число как процент

    Args:
        value: Значение для форматирования (0.85 -> 85%)
        decimal_places: Количество знаков после запятой

    Returns:
        Отформатированная строка в процентах
    """
    percentage = value * 100
    return f"{percentage:.{decimal_places}f}%"


def validate_input(
    data: Dict[str, Any], feature_ranges: Optional[Dict[str, tuple]] = None
) -> Dict[str, Any]:
    """
    Валидация входных данных для предсказания

    Args:
        data: Словарь с данными для валидации
        feature_ranges: Допустимые диапазоны для каждого признака

    Returns:
        Валидированные данные

    Raises:
        ValueError: Если данные не прошли валидацию
    """
    validated_data = data.copy()

    # Базовая валидация на наличие всех признаков
    required_features = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    for feature in required_features:
        if feature not in validated_data:
            raise ValueError(f"Отсутствует обязательный признак: {feature}")

        # Проверка на числовые значения
        try:
            validated_data[feature] = float(validated_data[feature])
        except (TypeError, ValueError):
            raise ValueError(f"Признак {feature} должен быть числом")

    # Дополнительная валидация диапазонов
    if feature_ranges:
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature in validated_data:
                value = validated_data[feature]
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"Значение {feature} ({value}) выходит за допустимые пределы [{min_val}, {max_val}]"
                    )

    return validated_data


def create_prediction_summary(
    prediction: float, avg_price: float, features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Создает сводку по предсказанию

    Args:
        prediction: Предсказанная цена
        avg_price: Средняя цена
        features: Признаки, использованные для предсказания

    Returns:
        Словарь со сводкой
    """
    price_dollars = prediction * 100_000
    avg_price_dollars = avg_price * 100_000
    price_diff = price_dollars - avg_price_dollars

    return {
        "predicted_price": price_dollars,
        "formatted_price": format_currency(price_dollars),
        "average_price": avg_price_dollars,
        "formatted_average": format_currency(avg_price_dollars),
        "price_difference": price_diff,
        "formatted_difference": format_currency(abs(price_diff)),
        "is_above_average": price_diff > 0,
        "percentage_difference": abs(price_diff) / avg_price_dollars * 100,
        "features_used": features,
    }
