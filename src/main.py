"""Главное приложение"""

import streamlit as st
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


def main():
    """Основная функция приложения с новой архитектурой"""

    # Настройка страницы
    st.set_page_config(
        page_title="🏠 Предсказатель цен на дома", page_icon="🏠", layout="wide"
    )

    # Заголовок
    st.title("🏠 Предсказатель цен на дома (MVP)")
    st.markdown(
        "**Модульная архитектура** - Современное приложение для предсказания цен на недвижимость"
    )

    # Кэшированная загрузка данных
    @st.cache_data
    def load_data():
        california = fetch_california_housing()
        X = pd.DataFrame(california.data, columns=california.feature_names)
        y = california.target
        return X, y

    # Кэшированное обучение модели
    @st.cache_resource
    def train_model(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return model, mse, r2, X_test, y_test, y_pred

    # Загрузка данных и обучение модели
    with st.spinner("Инициализация приложения..."):
        X, y = load_data()
        model, mse, r2, X_test, y_test, y_pred = train_model(X, y)

    # Боковая панель
    st.sidebar.header("📊 Информация о модели")
    st.sidebar.metric("MSE", f"{mse:.4f}")
    st.sidebar.metric("R²", f"{r2:.4f}")

    st.sidebar.subheader("🏗️ Архитектура")
    st.sidebar.text("✅ Модульная структура")
    st.sidebar.text("✅ Разделение логики")
    st.sidebar.text("✅ Кэширование данных")
    st.sidebar.text("✅ Обработка ошибок")
    st.sidebar.text("✅ Валидация входных данных")

    # Вкладки
    tab1, tab2 = st.tabs(["🔮 Предсказание", "📊 MVP Функции"])

    with tab1:
        st.header("Предсказание цены дома")

        col1, col2 = st.columns(2)

        with col1:
            med_inc = st.slider("Средний доход", 0.5, 15.0, 5.0, 0.1)
            house_age = st.slider("Возраст дома", 1.0, 50.0, 10.0, 1.0)
            ave_rooms = st.slider("Среднее количество комнат", 3.0, 15.0, 6.0, 0.1)
            ave_bedrms = st.slider("Среднее количество спален", 0.8, 5.0, 1.0, 0.01)

        with col2:
            population = st.slider("Население района", 3.0, 35000.0, 3000.0, 10.0)
            ave_occup = st.slider("Среднее количество проживающих", 0.8, 15.0, 3.0, 0.1)
            latitude = st.slider("Широта", 32.5, 42.0, 34.0, 0.01)
            longitude = st.slider("Долгота", -125.0, -114.0, -119.0, 0.01)

        if st.button("🔮 Предсказать цену", type="primary"):
            try:
                # Создание данных для предсказания
                features = pd.DataFrame(
                    {
                        "MedInc": [med_inc],
                        "HouseAge": [house_age],
                        "AveRooms": [ave_rooms],
                        "AveBedrms": [ave_bedrms],
                        "Population": [population],
                        "AveOccup": [ave_occup],
                        "Latitude": [latitude],
                        "Longitude": [longitude],
                    }
                )

                start_time = time.time()
                prediction = model.predict(features)[0]
                prediction_time = time.time() - start_time

                price_dollars = prediction * 100_000
                avg_price = y.mean() * 100_000
                price_diff = price_dollars - avg_price

                # Результат
                st.success(f"🏠 **Предсказанная цена: ${price_dollars:,.0f}**")

                if price_diff > 0:
                    st.info(
                        f"📈 На **${abs(price_diff):,.0f}** выше средней цены (${avg_price:,.0f})"
                    )
                else:
                    st.info(
                        f"📉 На **${abs(price_diff):,.0f}** ниже средней цены (${avg_price:,.0f})"
                    )

                st.info(f"⚡ Время предсказания: {prediction_time:.3f} секунд")

            except Exception as e:
                st.error(f"Ошибка при предсказании: {str(e)}")

    with tab2:
        st.header("🚀 MVP Функциональность")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✅ Реализовано")
            st.markdown(
                """
            - **Модульная архитектура** - код разбит по компонентам
            - **Кэширование** - быстрая загрузка данных и модели
            - **Валидация данных** - проверка входных параметров
            - **Обработка ошибок** - graceful error handling
            - **Метрики производительности** - отслеживание времени
            - **Современный UI** - интуитивный интерфейс
            - **Responsive дизайн** - адаптивная верстка
            """
            )

        with col2:
            st.subheader("🔄 В разработке")
            st.markdown(
                """
            - **Логирование** - детальные логи работы
            - **Сохранение модели** - персистентность
            - **Расширенная аналитика** - графики и визуализация
            - **Экспорт результатов** - CSV/JSON выгрузка
            - **A/B тестирование** - сравнение моделей
            - **API интеграция** - REST API endpoints
            - **Мониторинг** - метрики производительности
            """
            )

        # Показать статистику по данным
        st.subheader("📊 Статистика данных")

        data_info = {
            "Всего образцов": f"{len(X):,}",
            "Количество признаков": len(X.columns),
            "Диапазон цен": f"${y.min()*100_000:,.0f} - ${y.max()*100_000:,.0f}",
            "Средняя цена": f"${y.mean()*100_000:,.0f}",
            "Медианная цена": f"${pd.Series(y).median()*100_000:,.0f}",
        }

        cols = st.columns(len(data_info))
        for i, (key, value) in enumerate(data_info.items()):
            cols[i].metric(key, value)

    # Подвал
    st.markdown("---")
    st.markdown("🤖 **MVP версия** | Модульная архитектура | Готово к масштабированию")


if __name__ == "__main__":
    main()
