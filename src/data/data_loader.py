"""Модуль для загрузки и обработки данных"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from typing import Tuple, Dict, Any

from ..config.settings import get_feature_config


class DataLoader:
    """Класс для загрузки и предобработки данных"""

    def __init__(self):
        self.feature_config = get_feature_config()
        self.X = None
        self.y = None
        self.data_loaded = False

    def load_california_housing(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Загрузка данных California Housing Dataset

        Returns:
            Кортеж (признаки, целевая переменная)
        """
        california = fetch_california_housing()

        # Создание DataFrame с признаками
        self.X = pd.DataFrame(california.data, columns=california.feature_names)
        self.y = california.target

        self.data_loaded = True

        return self.X, self.y

    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Получение статистики по признакам

        Returns:
            Словарь со статистикой для каждого признака
        """
        if not self.data_loaded:
            raise ValueError(
                "Данные не загружены. Сначала вызовите load_california_housing()"
            )

        stats = {}
        for column in self.X.columns:
            stats[column] = {
                "mean": float(self.X[column].mean()),
                "std": float(self.X[column].std()),
                "min": float(self.X[column].min()),
                "max": float(self.X[column].max()),
                "median": float(self.X[column].median()),
                "q25": float(self.X[column].quantile(0.25)),
                "q75": float(self.X[column].quantile(0.75)),
            }

        return stats

    def get_target_statistics(self) -> Dict[str, float]:
        """
        Получение статистики по целевой переменной

        Returns:
            Словарь со статистикой целевой переменной
        """
        if not self.data_loaded:
            raise ValueError(
                "Данные не загружены. Сначала вызовите load_california_housing()"
            )

        return {
            "mean": float(self.y.mean()),
            "std": float(self.y.std()),
            "min": float(self.y.min()),
            "max": float(self.y.max()),
            "median": float(np.median(self.y)),
            "q25": float(np.quantile(self.y, 0.25)),
            "q75": float(np.quantile(self.y, 0.75)),
        }

    def get_correlations(self) -> pd.Series:
        """
        Получение корреляций признаков с целевой переменной

        Returns:
            Series с корреляциями
        """
        if not self.data_loaded:
            raise ValueError(
                "Данные не загружены. Сначала вызовите load_california_housing()"
            )

        y_series = pd.Series(self.y, index=self.X.index, name="Target")
        return self.X.corrwith(y_series).sort_values(ascending=False)

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Получение общей сводки о данных

        Returns:
            Словарь с общей информацией о данных
        """
        if not self.data_loaded:
            return {"status": "Данные не загружены"}

        return {
            "status": "Данные загружены",
            "samples_count": len(self.X),
            "features_count": len(self.X.columns),
            "feature_names": list(self.X.columns),
            "target_range": {
                "min": float(self.y.min()),
                "max": float(self.y.max()),
                "mean": float(self.y.mean()),
            },
            "missing_values": self.X.isnull().sum().to_dict(),
        }

    def validate_prediction_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Валидация данных для предсказания

        Args:
            data: Словарь с данными для предсказания

        Returns:
            Валидированные данные

        Raises:
            ValueError: Если данные не прошли валидацию
        """
        if not self.data_loaded:
            raise ValueError(
                "Данные не загружены. Сначала вызовите load_california_housing()"
            )

        validated_data = data.copy()

        # Проверка наличия всех необходимых признаков
        required_features = list(self.X.columns)
        missing_features = [f for f in required_features if f not in validated_data]

        if missing_features:
            raise ValueError(f"Отсутствуют признаки: {missing_features}")

        # Валидация и приведение типов
        for feature in required_features:
            try:
                validated_data[feature] = float(validated_data[feature])
            except (TypeError, ValueError):
                raise ValueError(f"Признак {feature} должен быть числом")

            # Проверка диапазонов
            min_val = self.X[feature].min()
            max_val = self.X[feature].max()

            if not (min_val <= validated_data[feature] <= max_val):
                raise ValueError(
                    f"Значение {feature} ({validated_data[feature]}) "
                    f"выходит за пределы [{min_val:.3f}, {max_val:.3f}]"
                )

        return validated_data
