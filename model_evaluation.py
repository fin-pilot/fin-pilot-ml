import itertools
import os
import warnings

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# =========================================================
# ЗАВАНТАЖЕННЯ ДАТАСЕТУ
# =========================================================
print("Завантаження датасету з Kaggle...")

path = kagglehub.dataset_download("ramyapintchy/personal-finance-data")

csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
csv_file = csv_files[0]

file_path = os.path.join(path, csv_file)

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# =========================================================
# ПІДГОТОВКА ДАНИХ
# =========================================================
date_col = "Date"
amount_col = "Amount"
type_col = "Transaction Type" if "Transaction Type" in df.columns else "Type"

df[date_col] = pd.to_datetime(
    df[date_col],
    format="mixed",
    errors="coerce",
)

df = df.dropna(subset=[date_col])

# Беремо лише витрати
expenses_df = df[
    df[type_col].astype(str).str.contains("Debit|Expense", case=False, na=False)
].copy()

expenses_df.set_index(date_col, inplace=True)

# =========================================================
# WEEKLY RESAMPLING
# =========================================================
weekly_expenses = expenses_df[amount_col].resample("W").sum().fillna(0)

# Прибираємо початкові нулі
start_date = weekly_expenses[weekly_expenses > 0].index[0]
weekly_expenses = weekly_expenses.loc[start_date:]

print(f"\nКількість weekly observations: {len(weekly_expenses)}")

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
test_size = max(4, int(len(weekly_expenses) * 0.20))

train_data = weekly_expenses.iloc[:-test_size]
test_data = weekly_expenses.iloc[-test_size:]

print(f"Train size: {len(train_data)}")
print(f"Test size : {len(test_data)}")

# =========================================================
# GRID SEARCH
# =========================================================
print("\nЗапуск Grid Search для SARIMA...")

# Невеликий search space для стабільності
p = range(0, 3)
d = range(0, 2)
q = range(0, 3)

P = range(0, 2)
D = range(0, 2)
Q = range(0, 2)

# Weekly seasonality
SEASONAL_PERIOD = 52

pdq = list(itertools.product(p, d, q))
seasonal_pdq = list(itertools.product(P, D, Q, [SEASONAL_PERIOD]))

total_combinations = len(pdq) * len(seasonal_pdq)

print(f"Буде перевірено моделей: {total_combinations}")

best_aic = np.inf
best_order = None
best_seasonal_order = None

current_iteration = 0
failed_models = 0

for param in pdq:
    for seasonal_param in seasonal_pdq:

        current_iteration += 1

        if current_iteration % 25 == 0:
            print(
                f"Прогрес: {current_iteration}/{total_combinations} | "
                f"Best AIC: {best_aic:.2f}"
            )

        try:
            model = SARIMAX(
                train_data,
                order=param,
                seasonal_order=seasonal_param,
                enforce_stationarity=True,
                enforce_invertibility=True,
            )

            results = model.fit(disp=False)

            # Перевірка convergence
            if not results.mle_retvals.get("converged", False):
                continue

            # Перевірка AIC
            if np.isnan(results.aic) or np.isinf(results.aic):
                continue

            # Garbage models filter
            if results.aic < 10:
                continue

            if results.aic < best_aic:
                best_aic = results.aic
                best_order = param
                best_seasonal_order = seasonal_param

                print(
                    f"\nНова найкраща модель:"
                    f"\nOrder: {best_order}"
                    f"\nSeasonal: {best_seasonal_order}"
                    f"\nAIC: {best_aic:.2f}\n"
                )

        except Exception:
            failed_models += 1
            continue

print("\n========================================")
print("GRID SEARCH ЗАВЕРШЕНО")
print("========================================")

print(f"Failed models: {failed_models}")

print(
    f"\nНайкращі параметри:"
    f"\nOrder: {best_order}"
    f"\nSeasonal Order: {best_seasonal_order}"
    f"\nBest AIC: {best_aic:.2f}"
)

# =========================================================
# FINAL SARIMA MODEL
# =========================================================
print("\nНавчання фінальної SARIMA моделі...")

sarima_model = SARIMAX(
    train_data,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=True,
    enforce_invertibility=True,
)

sarima_result = sarima_model.fit(disp=False)

sarima_forecast = sarima_result.forecast(steps=len(test_data))

sarima_forecast.index = test_data.index

# =========================================================
# HOLT-WINTERS
# =========================================================
print("\nНавчання Holt-Winters моделі...")

hw_model = ExponentialSmoothing(
    train_data,
    trend="add",
    seasonal="add",
    seasonal_periods=SEASONAL_PERIOD,
    initialization_method="estimated",
)

hw_result = hw_model.fit()

hw_forecast = hw_result.forecast(steps=len(test_data))

# =========================================================
# EVALUATION
# =========================================================
models = {
    "SARIMA": sarima_forecast,
    "Holt-Winters": hw_forecast,
}

metrics = {}

for model_name, forecast in models.items():

    mae = mean_absolute_error(test_data, forecast)

    rmse = np.sqrt(mean_squared_error(test_data, forecast))

    wape = (np.sum(np.abs(test_data - forecast)) / np.sum(test_data)) * 100

    metrics[model_name] = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "WAPE (%)": round(wape, 2),
    }

print("\n========================================")
print("МЕТРИКИ ЯКОСТІ")
print("========================================")

metrics_df = pd.DataFrame(metrics).T
print(metrics_df)

# =========================================================
# PLOT
# =========================================================
plt.figure(figsize=(14, 7))

plt.plot(
    train_data.index,
    train_data,
    label="Train",
    marker="o",
    alpha=0.7,
)

plt.plot(
    test_data.index,
    test_data,
    label="Actual",
    marker="o",
    linewidth=2,
)

plt.plot(
    test_data.index,
    sarima_forecast,
    label="SARIMA Forecast",
    linestyle="--",
    linewidth=2,
)

plt.plot(
    test_data.index,
    hw_forecast,
    label="Holt-Winters Forecast",
    linestyle="-.",
    linewidth=2,
)

plt.title("Weekly Expense Forecasting")
plt.xlabel("Date")
plt.ylabel("Expenses")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

output_file = "weekly_expense_forecast.png"

plt.savefig(output_file, dpi=300)

print(f"\nГрафік збережено: {output_file}")

plt.show()
