from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pmdarima as pm
from pmdarima.arima import ARIMA
from statsmodels.tsa.stattools import adfuller

from fin_pilot_ml.config import MLSettings


class TransactionForecaster:
    def __init__(self, config: MLSettings) -> None:
        self.config = config
        self.sarima_config = self.config.forecaster.sarima

        print("Initializing transaction forecaster.")

        self.model: ARIMA | None = None

    def fit(self, series: pd.Series) -> None:
        print("Starting SARIMA parameter search...")

        prepared_series = self._prepare_series(series)

        if len(prepared_series) < 14:
            raise ValueError(
                f"Minimum 14 observations required, got {len(prepared_series)}."
            )

        if prepared_series.nunique() <= 1:
            raise ValueError(
                "Series is constant. SARIMA cannot be trained on constant data."
            )

        seasonal_enabled = self.sarima_config.seasonal
        seasonal_period = self.sarima_config.seasonal_period

        min_required = seasonal_period * 2

        if seasonal_enabled and len(prepared_series) < min_required:
            print(
                (
                    "Series too short for seasonal SARIMA "
                    f"(len={len(prepared_series)}, required={min_required}). "
                    "Falling back to non-seasonal ARIMA."
                )
            )

            seasonal_enabled = False
            seasonal_period = 0

        try:
            auto_arima_kwargs = {
                "seasonal": seasonal_enabled,
                "m": seasonal_period if seasonal_enabled else 1,
                "start_p": 0,
                "start_q": 0,
                "max_p": self.sarima_config.max_p,
                "max_q": self.sarima_config.max_q,
                "max_d": self.sarima_config.max_d,
                "start_P": 0,
                "start_Q": 0,
                "max_P": self.sarima_config.max_P,
                "max_Q": self.sarima_config.max_Q,
                "max_D": self.sarima_config.max_D,
                "stepwise": self.sarima_config.stepwise,
                "trace": self.sarima_config.trace,
                "error_action": self.sarima_config.error_action,
                "suppress_warnings": self.sarima_config.suppress_warnings,
                "information_criterion": (
                    self.sarima_config.information_criterion
                ),
                "with_intercept": "auto",
                "stationary": False,
                "n_jobs": self.sarima_config.n_jobs,
            }

            if self.sarima_config.max_order is not None:
                auto_arima_kwargs["max_order"] = self.sarima_config.max_order

            self.model = pm.auto_arima(
                prepared_series,
                **auto_arima_kwargs,
            )

        except ValueError as error:
            print(
                f"SARIMA fitting failed ({error}). "
                "Retrying with simpler non-seasonal ARIMA."
            )

            fallback_kwargs = {
                "seasonal": False,
                "m": 1,
                "start_p": 0,
                "start_q": 0,
                "max_p": self.sarima_config.max_p,
                "max_q": self.sarima_config.max_q,
                "max_d": self.sarima_config.max_d,
                "stepwise": self.sarima_config.stepwise,
                "trace": self.sarima_config.trace,
                "error_action": self.sarima_config.error_action,
                "suppress_warnings": self.sarima_config.suppress_warnings,
                "with_intercept": "auto",
                "stationary": False,
                "n_jobs": self.sarima_config.n_jobs,
            }

            if self.sarima_config.max_order is not None:
                fallback_kwargs["max_order"] = self.sarima_config.max_order

            self.model = pm.auto_arima(
                prepared_series,
                **fallback_kwargs,
            )

        if self.model is None:
            raise ValueError("SARIMA model fitting failed.")

        print(
            "Model fitted successfully. "
            f"Order={self.model.order} "
            f"Seasonal={self.model.seasonal_order}"
        )

    def update(self, new_series: pd.Series) -> None:
        if self.model is None:
            raise ValueError("Model is not initialized.")

        print("Updating forecasting model...")

        prepared_series = self._prepare_series(new_series)

        self.model.update(prepared_series)

    def forecast(self, steps: int = 30) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model is not initialized.")

        forecast_vals, conf_int = self.model.predict(
            n_periods=steps,
            return_conf_int=True,
        )

        return pd.DataFrame(
            {
                "predicted_y": forecast_vals.astype(float),
                "conf_lower": conf_int[:, 0].astype(float),
                "conf_upper": conf_int[:, 1].astype(float),
            }
        )

    def save_model(self) -> None:
        model_path = self._model_path

        model_path.parent.parent.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving forecasting model to {model_path}")

        joblib.dump(self.model, model_path)

    def load_model(self) -> None:
        model_path = self._model_path

        if not model_path.exists():
            print(f"Forecasting model not found: {model_path}")
            return

        print(f"Loading forecasting model from {model_path}")

        self.model = joblib.load(model_path)

    def check_stationarity(self, series: pd.Series) -> bool:
        prepared_series = self._prepare_series(series)

        if len(prepared_series) < 15:
            return True

        result: tuple[Any, ...] = adfuller(prepared_series.dropna())

        p_value = float(result[1])

        is_stationary = bool(p_value < 0.05)

        print(f"ADF p-value: {p_value:.4f}")
        print(f"Stationary: {is_stationary}")

        return is_stationary

    def _prepare_series(self, series: pd.Series) -> pd.Series:
        prepared = series.copy()

        prepared.index = pd.to_datetime(prepared.index)

        prepared = prepared.sort_index()

        prepared = prepared.astype(float)

        prepared = prepared.fillna(0.0)

        return prepared

    @property
    def _model_path(self) -> Path:
        return Path(self.config.forecaster.model.path)
