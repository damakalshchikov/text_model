"""Модель для предсказания цен на дома"""

import pickle
import time
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from ..config.settings import get_model_config, get_feature_config


class HousePriceModel:
    """Класс для работы с моделью предсказания цен на дома"""

    def __init__(self):
        self.config = get_model_config()
        self.feature_config = get_feature_config()
        self.model = None
        self.is_trained = False
        self.training_metrics = {}
        self.feature_ranges = {}

    def create_model(self) -> RandomForestRegressor:
        """Создание модели с параметрами из конфигурации"""
        return RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            n_jobs=-1,  # Используем все доступные ядра
        )

    def train(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Обучение модели

        Args:
            X: Признаки
            y: Целевая переменная

        Returns:
            Словарь с метриками качества модели
        """
        start_time = time.time()

        # Сохранение диапазонов признаков для валидации
        self.feature_ranges = {col: (X[col].min(), X[col].max()) for col in X.columns}

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.train_random_state,
        )

        # Создание и обучение модели
        self.model = self.create_model()
        self.model.fit(X_train, y_train)

        # Предсказания
        y_pred = self.model.predict(X_test)

        # Метрики качества
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Кросс-валидация
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=self.config.cv_folds, scoring="r2"
        )

        training_time = time.time() - start_time

        # Сохранение метрик
        self.training_metrics = {
            "mse": mse,
            "r2": r2,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "training_time": training_time,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        self.is_trained = True

        return self.training_metrics

    def predict(self, features: Dict[str, Any]) -> float:
        """
        Предсказание цены для одного дома

        Args:
            features: Словарь с признаками дома

        Returns:
            Предсказанная цена

        Raises:
            ValueError: Если модель не обучена или неверные данные
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите метод train()")

        # Создание DataFrame для предсказания
        df = pd.DataFrame([features])

        # Предсказание
        prediction = self.model.predict(df)[0]

        return prediction

    def predict_batch(self, features_list: pd.DataFrame) -> np.ndarray:
        """
        Предсказание цен для множества домов

        Args:
            features_list: DataFrame с признаками домов

        Returns:
            Массив предсказанных цен
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите метод train()")

        return self.model.predict(features_list)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Получение важности признаков

        Returns:
            Словарь с важностью признаков
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Сначала вызовите метод train()")

        feature_names = self.model.feature_names_in_
        importances = self.model.feature_importances_

        return dict(zip(feature_names, importances))

    def save_model(self, filepath: str) -> None:
        """
        Сохранение обученной модели

        Args:
            filepath: Путь для сохранения модели
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена. Нечего сохранять.")

        model_data = {
            "model": self.model,
            "training_metrics": self.training_metrics,
            "feature_ranges": self.feature_ranges,
            "config": self.config,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str) -> None:
        """
        Загрузка обученной модели

        Args:
            filepath: Путь к файлу модели
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.training_metrics = model_data["training_metrics"]
        self.feature_ranges = model_data["feature_ranges"]
        self.is_trained = True

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Получение сводки о модели

        Returns:
            Словарь с информацией о модели
        """
        if not self.is_trained:
            return {"status": "Модель не обучена"}

        return {
            "status": "Модель обучена",
            "model_type": "Random Forest Regressor",
            "n_estimators": self.config.n_estimators,
            "training_metrics": self.training_metrics,
            "feature_count": len(self.feature_ranges),
            "feature_names": list(self.feature_ranges.keys()),
        }
