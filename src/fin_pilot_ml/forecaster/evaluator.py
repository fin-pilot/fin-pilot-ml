from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(slots=True)
class ForecastMetrics:
    mae: float
    rmse: float
    wape: float


class ForecastEvaluator:
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ForecastMetrics:
        print("Evaluating forecasting model...")

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        total_actual = np.sum(np.abs(y_true))

        wape = (
            float(np.sum(np.abs(y_true - y_pred)) / total_actual)
            if total_actual > 0
            else 0.0
        )

        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"WAPE : {wape:.4f}")

        return ForecastMetrics(mae=mae, rmse=rmse, wape=wape)

    def evaluate_walk_forward(
        self,
        model_factory: Callable[[], object],
        series: pd.Series,
        initial_train_size: int = 52,
        step: int = 4,
    ) -> ForecastMetrics:
        print("Starting walk-forward validation...")

        if len(series) <= initial_train_size:
            print("Series too short for walk-forward validation.")

            return ForecastMetrics(
                mae=0.0,
                rmse=0.0,
                wape=0.0,
            )

        all_true: list[float] = []
        all_pred: list[float] = []

        for i in range(initial_train_size, len(series), step):
            train_split = series.iloc[:i]
            test_split = series.iloc[i : i + step]

            try:
                model = model_factory()

                model.fit(train_split)

                forecast_df: pd.DataFrame = model.forecast(
                    steps=len(test_split)
                )

                all_true.extend(test_split.values.tolist())
                all_pred.extend(forecast_df["predicted_y"].values.tolist())

            except Exception as error:
                print(f"Walk-forward failed at step {i}: {error}")

        if not all_true or not all_pred:
            print("No successful walk-forward predictions generated.")

            return ForecastMetrics(
                mae=0.0,
                rmse=0.0,
                wape=0.0,
            )

        return self.evaluate(
            np.array(all_true),
            np.array(all_pred),
        )
