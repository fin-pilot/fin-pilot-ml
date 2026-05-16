import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from fin_pilot_ml.forecasting.plots import (
    ForecastPlots,
)
from fin_pilot_ml.forecasting.schemas import (
    ForecastMetrics,
)


class ForecastEvaluator:
    @staticmethod
    def evaluate(
        actual: pd.Series,
        predicted: pd.Series,
        plots_dir: Path,
        model_name: str,
    ) -> ForecastMetrics:

        plots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        residuals = actual - predicted

        mae = float(
            mean_absolute_error(
                actual,
                predicted,
            )
        )

        mse = float(
            mean_squared_error(
                actual,
                predicted,
            )
        )

        rmse = float(np.sqrt(mse))

        denominator = np.where(
            actual == 0,
            1e-8,
            actual,
        )

        mape = float(np.mean(np.abs((actual - predicted) / denominator)) * 100)

        smape = float(
            (
                np.mean(
                    2
                    * np.abs(predicted - actual)
                    / (np.abs(actual) + np.abs(predicted) + 1e-8)
                )
            )
            * 100
        )

        wape = float(
            (np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual))) * 100
        )

        r2 = float(
            r2_score(
                actual,
                predicted,
            )
        )

        mbe = float(np.mean(predicted - actual))

        metrics = ForecastMetrics(
            mae=round(mae, 4),
            mse=round(mse, 4),
            rmse=round(rmse, 4),
            mape=round(mape, 4),
            smape=round(smape, 4),
            wape=round(wape, 4),
            r2=round(r2, 4),
            mbe=round(mbe, 4),
            residual_mean=round(
                float(residuals.mean()),
                4,
            ),
            residual_std=round(
                float(residuals.std()),
                4,
            ),
            plots_dir=plots_dir,
        )

        ForecastPlots.actual_vs_forecast(
            actual,
            predicted,
            plots_dir / f"{model_name}_forecast.png",
        )

        ForecastPlots.residual_plot(
            residuals,
            plots_dir / f"{model_name}_residuals.png",
        )

        ForecastPlots.residual_distribution(
            residuals,
            plots_dir / f"{model_name}_residual_distribution.png",
        )

        ForecastPlots.rolling_error_plot(
            actual,
            predicted,
            plots_dir / f"{model_name}_rolling_error.png",
        )

        with open(
            plots_dir / f"{model_name}_metrics.json",
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metrics.__dict__,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return metrics
