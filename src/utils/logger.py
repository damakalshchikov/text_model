"""Утилиты для логирования"""

import logging
import os
from typing import Optional


def setup_logger(
    name: str = "house_price_app", log_file: Optional[str] = None, level: str = "INFO"
) -> logging.Logger:
    """
    Настройка логгера для приложения

    Args:
        name: Имя логгера
        log_file: Путь к файлу логов (опционально)
        level: Уровень логирования

    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)

    # Избегаем дублирования обработчиков
    if logger.handlers:
        return logger

    # Установка уровня логирования
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Формат сообщений
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Файловый обработчик (если указан файл)
    if log_file:
        # Создание директории для логов, если она не существует
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "house_price_app") -> logging.Logger:
    """
    Получить существующий логгер

    Args:
        name: Имя логгера

    Returns:
        Логгер
    """
    return logging.getLogger(name)


def log_prediction(
    logger: logging.Logger,
    features: dict,
    prediction: float,
    execution_time: Optional[float] = None,
) -> None:
    """
    Логирование предсказания

    Args:
        logger: Логгер
        features: Признаки для предсказания
        prediction: Результат предсказания
        execution_time: Время выполнения предсказания
    """
    message = f"Prediction made: {prediction:.2f} for features: {features}"
    if execution_time:
        message += f" (execution time: {execution_time:.3f}s)"

    logger.info(message)


def log_model_training(
    logger: logging.Logger,
    model_name: str,
    mse: float,
    r2: float,
    training_time: Optional[float] = None,
) -> None:
    """
    Логирование обучения модели

    Args:
        logger: Логгер
        model_name: Название модели
        mse: Среднеквадратичная ошибка
        r2: Коэффициент детерминации
        training_time: Время обучения
    """
    message = f"Model '{model_name}' trained: MSE={mse:.4f}, R²={r2:.4f}"
    if training_time:
        message += f" (training time: {training_time:.3f}s)"

    logger.info(message)


def log_error(logger: logging.Logger, error: Exception, context: str = "") -> None:
    """
    Логирование ошибки

    Args:
        logger: Логгер
        error: Исключение
        context: Контекст ошибки
    """
    error_message = (
        f"Error in {context}: {str(error)}" if context else f"Error: {str(error)}"
    )
    logger.error(error_message, exc_info=True)
