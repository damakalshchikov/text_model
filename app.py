import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# Настройка страницы
st.set_page_config(
    page_title="Предсказатель цен на дома", page_icon="🏠", layout="wide"
)

# Заголовок приложения
st.title("🏠 Предсказатель цен на дома")
st.markdown(
    "Простое приложение для предсказания цен на недвижимость с использованием машинного обучения"
)


# Кэширование данных и модели
@st.cache_data
def load_data():
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = california.target
    return X, y


@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Метрики
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2, X_test, y_test, y_pred


# Загрузка данных и обучение модели
with st.spinner("Загружаем данные и обучаем модель..."):
    X, y = load_data()
    model, mse, r2, X_test, y_test, y_pred = train_model(X, y)

# Боковая панель с информацией о модели
st.sidebar.header("📊 Информация о модели")
st.sidebar.metric("Среднеквадратичная ошибка (MSE)", f"{mse:.4f}")
st.sidebar.metric("Коэффициент детерминации R²", f"{r2:.4f}")

# Описание признаков
feature_descriptions = {
    "MedInc": "Средний доход в районе",
    "HouseAge": "Средний возраст домов в районе",
    "AveRooms": "Среднее количество комнат на дом",
    "AveBedrms": "Среднее количество спален на дом",
    "Population": "Население района",
    "AveOccup": "Среднее количество проживающих на дом",
    "Latitude": "Географическая широта района",
    "Longitude": "Географическая долгота района",
}

# Создание вкладок
tab1, tab2, tab3 = st.tabs(
    ["🔮 Предсказание", "📈 Анализ данных", "📊 Результаты модели"]
)

with tab1:
    st.header("Введите характеристики дома для предсказания цены")

    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.slider(
            "Средний доход в районе",
            min_value=float(X["MedInc"].min()),
            max_value=float(X["MedInc"].max()),
            value=float(X["MedInc"].mean()),
            step=0.1,
        )

        house_age = st.slider(
            "Возраст дома",
            min_value=float(X["HouseAge"].min()),
            max_value=float(X["HouseAge"].max()),
            value=float(X["HouseAge"].mean()),
            step=1.0,
        )

        ave_rooms = st.slider(
            "Среднее количество комнат",
            min_value=float(X["AveRooms"].min()),
            max_value=float(X["AveRooms"].max()),
            value=float(X["AveRooms"].mean()),
            step=0.1,
        )

        ave_bedrms = st.slider(
            "Среднее количество спален",
            min_value=float(X["AveBedrms"].min()),
            max_value=float(X["AveBedrms"].max()),
            value=float(X["AveBedrms"].mean()),
            step=0.01,
        )

    with col2:
        population = st.slider(
            "Население района",
            min_value=float(X["Population"].min()),
            max_value=float(X["Population"].max()),
            value=float(X["Population"].mean()),
            step=10.0,
        )

        ave_occup = st.slider(
            "Среднее количество проживающих",
            min_value=float(X["AveOccup"].min()),
            max_value=float(X["AveOccup"].max()),
            value=float(X["AveOccup"].mean()),
            step=0.1,
        )

        latitude = st.slider(
            "Широта",
            min_value=float(X["Latitude"].min()),
            max_value=float(X["Latitude"].max()),
            value=float(X["Latitude"].mean()),
            step=0.01,
        )

        longitude = st.slider(
            "Долгота",
            min_value=float(X["Longitude"].min()),
            max_value=float(X["Longitude"].max()),
            value=float(X["Longitude"].mean()),
            step=0.01,
        )

    # Кнопка для предсказания
    if st.button("🔮 Предсказать цену", type="primary"):
        # Создание DataFrame для предсказания
        input_data = pd.DataFrame(
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

        # Предсказание
        prediction = model.predict(input_data)[0]
        price_dollars = prediction * 100_000

        # Отображение результата
        st.success(f"🏠 **Предсказанная цена дома: ${price_dollars:,.2f}**")

        # Дополнительная информация
        avg_price = y.mean() * 100_000
        price_diff = price_dollars - avg_price

        if price_diff > 0:
            st.info(
                f"📈 Эта цена на {price_diff:,.0f} выше средней цены {avg_price:,.0f}"
            )
        else:
            st.info(
                f"📉 Эта цена на {abs(price_diff):,.0f} ниже средней цены {avg_price:,.0f}"
            )

with tab2:
    st.header("Анализ данных")

    # Основная статистика
    st.subheader("Основная статистика")
    st.dataframe(X.describe())

    # Корреляция
    st.subheader("Корреляция признаков с ценой")
    fig, ax = plt.subplots(figsize=(10, 8))
    y_series = pd.Series(y, index=X.index, name="Target")
    correlations = X.corrwith(y_series).sort_values(ascending=False)
    sns.barplot(x=correlations.values, y=correlations.index, ax=ax)
    ax.set_xlabel("Корреляция с ценой")
    ax.set_title("Корреляция признаков с ценой дома")
    st.pyplot(fig)

    # Распределение цен
    st.subheader("Распределение цен")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y * 100_000, bins=50, alpha=0.7, color="skyblue")
    ax.set_xlabel("Цена дома ($)")
    ax.set_ylabel("Количество домов")
    ax.set_title("Распределение цен на дома")
    st.pyplot(fig)

with tab3:
    st.header("Результаты модели")

    # График реальные vs предсказанные
    st.subheader("Реальные vs Предсказанные цены")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_test * 100_000, y_pred * 100_000, alpha=0.5)
    ax.plot(
        [y.min() * 100_000, y.max() * 100_000],
        [y.min() * 100_000, y.max() * 100_000],
        "r--",
    )
    ax.set_xlabel("Реальные цены ($)")
    ax.set_ylabel("Предсказанные цены ($)")
    ax.set_title("Точность предсказаний модели")
    st.pyplot(fig)

    # Важность признаков
    st.subheader("Важность признаков")
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x="importance", y="feature", ax=ax)
    ax.set_xlabel("Важность")
    ax.set_title("Важность признаков в модели")
    st.pyplot(fig)

# Подвал
st.markdown("---")
st.markdown("🤖 Приложение создано с использованием Streamlit и scikit-learn")
