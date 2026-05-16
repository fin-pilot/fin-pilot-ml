from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass(slots=True)
class TrainedModels:
    sarima: object
    holt_winters: object


class ForecastModels:
    @staticmethod
    def train_sarima(
        train_data: pd.Series,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
    ):
        model = SARIMAX(
            train_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )

        return model.fit(disp=False)

    @staticmethod
    def train_holt_winters(
        train_data: pd.Series,
        seasonal_period: int,
    ):
        model = ExponentialSmoothing(
            train_data,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_period,
            initialization_method="estimated",
        )

        return model.fit()
