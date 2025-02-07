import pandas as pd

# Загружаем данные с файла
file_path = "test.xlsx"  # Укажи файл с данными
df = pd.read_excel(file_path, engine="openpyxl")
df.columns = df.columns.str.strip()
df["Артикул"] = df["Артикул"].astype(str)
df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")
# Просмотр первых строк
print(df.head())  
print(df.info())  
print(df.columns)
print("Количество строк после фильтрации:", df.shape[0])
print(df["Артикул"].unique())
print(df["Дата"].dtype)
print(df["Дата"].head())
print(df[["Артикул", "Цена"]].head())


# Шаг 2: Очистка и фильтрация данных

# Задаем список артикулов
target_articles = ["905559214", "44403861", "95166060", "126373749", "304773881",
                   "440376223", "678167538", "811003631", "879714109", "-83054245"]

# Фильтруем по нужным артикулам
df = df[df["Артикул"].isin(target_articles)]

# Убираем пропущенные значения
df = df.dropna()

# Преобразуем столбец с датами в формат datetime
df["Дата"] = pd.to_datetime(df["Дата"])

# Добавляем колонку "Месяц-Год"
df["Месяц-Год"] = df["Дата"].dt.to_period("M")
df["Месяц-Год"] = df["Дата"].dt.to_period("M").astype(str)


# Группируем продажи по месяцам и артикулам
monthly_sales = df.groupby(["Месяц-Год", "Артикул"])["Количество"].sum().unstack().fillna(0)

print(monthly_sales.tail())
print("Размер monthly_sales:", monthly_sales.shape)
print(df[["Дата", "Месяц-Год"]].head())


# Шаг 3: Анализ 

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
for column in monthly_sales.columns:
    plt.plot(monthly_sales.index.astype(str), monthly_sales[column], label=column)

plt.title("Динамика продаж по месяцам")
plt.xlabel("Месяц")
plt.ylabel("Количество продаж")
plt.xticks(rotation=45)
plt.legend(title="Артикул")
plt.grid(True)
plt.show()

# Шаг 4: Прогнозирование продаж

from sklearn.linear_model import LinearRegression
import numpy as np

# Добавляем столбец "Месяц" как числовой индекс
monthly_sales["Месяц"] = np.arange(len(monthly_sales))

forecast = {}
forecast_revenue = {}  # Новый словарь для выручки

for article in target_articles:
    if article in monthly_sales.columns:
        X = monthly_sales[["Месяц"]].values.reshape(-1, 1)
        y = monthly_sales[article].values.reshape(-1, 1)

        if len(y) > 1:
            model = LinearRegression()
            model.fit(X, y)

            next_month = np.array([[X.max() + 1]])  # Следующий месяц
            prediction = model.predict(next_month)[0][0]
            forecast[article] = max(0, prediction)

            # Учитываем цену
            article_price = df[df["Артикул"] == article]["Цена"].mean()  # Средняя цена
            forecast_revenue[article] = forecast[article] * article_price
        else:
            forecast[article] = 0
            forecast_revenue[article] = 0

print("Прогноз по количеству:", forecast)
print("Прогноз по выручке:", forecast_revenue)


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Преобразуем прогноз в DataFrame
forecast_revenue_df = pd.DataFrame(forecast_revenue, index=["Прогноз выручки"]).T

plt.figure(figsize=(12, 6))
sns.barplot(x=forecast_revenue_df.index, y=forecast_revenue_df["Прогноз выручки"])
plt.title("Прогноз выручки на следующий месяц")
plt.xlabel("Артикул")
plt.ylabel("Прогнозируемая выручка")
plt.xticks(rotation=45)
plt.show()

forecast_df = pd.DataFrame(forecast, index=["Прогноз"]).T

plt.figure(figsize=(12, 6))
sns.barplot(x=forecast_df.index, y=forecast_df["Прогноз"])
plt.title("Прогноз продаж на следующий месяц")
plt.xlabel("Артикул")
plt.ylabel("Количество прогнозируемых продаж")
plt.xticks(rotation=45)
plt.show()

forecast_df = pd.DataFrame({
    "Артикул": forecast.keys(),
    "Прогноз продаж": forecast.values(),
    "Прогноз выручки": forecast_revenue.values()
})

print(forecast_df)  # Выведет в консоли таблицу

forecast_df.to_excel("результат_анализа.xlsx", index=False, engine="openpyxl")
print("Файл результат_анализа.xlsx сохранён!")
