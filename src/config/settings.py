from dataclasses import dataclass
from typing import Dict
from typing import Optional


@dataclass
class AppConfig:
    """Конфигурация приложения"""

    title: str = "🏠 Предсказатель цен на дома"
    page_icon: str = "🏠"
    layout: str = "wide"
    sidebar_title: str = "📊 Информация о модели"

    # Путь для сохранения модели
    model_save_path: str = "models/trained_model.pkl"

    # Логирование
    log_level: str = "INFO"
    log_file: str = "logs/app.log"


@dataclass
class ModelConfig:
    """Конфигурация модели машинного обучения"""

    # Параметры Random Forest
    n_estimators: int = 100
    random_state: int = 42
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1

    # Разделение данных
    test_size: float = 0.3
    train_random_state: int = 42

    # Валидация
    cv_folds: int = 5


@dataclass
class FeatureConfig:
    """Конфигурация признаков"""

    FEATURE_DESCRIPTIONS: Optional[Dict[str, str]] = None
    FEATURE_UNITS: Optional[Dict[str, str]] = None
    FEATURE_RANGES: Optional[Dict[str, tuple]] = None

    def __post_init__(self):
        self.FEATURE_DESCRIPTIONS = {
            "MedInc": "Средний доход в районе",
            "HouseAge": "Средний возраст домов в районе",
            "AveRooms": "Среднее количество комнат на дом",
            "AveBedrms": "Среднее количество спален на дом",
            "Population": "Население района",
            "AveOccup": "Среднее количество проживающих на дом",
            "Latitude": "Географическая широта района",
            "Longitude": "Географическая долгота района",
        }

        self.FEATURE_UNITS = {
            "MedInc": "десятки тысяч $",
            "HouseAge": "лет",
            "AveRooms": "комнат",
            "AveBedrms": "спален",
            "Population": "человек",
            "AveOccup": "человек/дом",
            "Latitude": "градусы",
            "Longitude": "градусы",
        }


# Глобальные экземпляры конфигурации
app_config = AppConfig()
model_config = ModelConfig()
feature_config = FeatureConfig()


def get_app_config() -> AppConfig:
    """Получить конфигурацию приложения"""
    return app_config


def get_model_config() -> ModelConfig:
    """Получить конфигурацию модели"""
    return model_config


def get_feature_config() -> FeatureConfig:
    """Получить конфигурацию признаков"""
    return feature_config
