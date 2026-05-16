from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml.forecaster.data_loader import ForecastingDataLoader
from ml.forecaster.model import TransactionForecaster
from ml.forecaster.preprocessing import ForecastPreprocessor
from ml.config import ml_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ForecastMetrics:
    mae: float
    rmse: float
    wape: float
    mape: float
    smape: float
    r2: float
    bias: float


class ForecastEvaluator:
    def evaluate(
        self,
        model: TransactionForecaster,
        test_series: pd.Series,
        actual_test_balance: pd.Series,
        last_train_balance: float,
        plot_dir: Path | None = None,
    ) -> ForecastMetrics:
        logger.info("Evaluating forecasting model on reconstructed balance...")

        # Model predicts the flows (not the balance)
        forecast_df = model.predict(steps=len(test_series))
        predicted_flows = forecast_df["predicted_y"].values

        # Reconstruct the balance using cumsum
        y_pred_balance = last_train_balance + np.cumsum(predicted_flows)
        y_true_balance = actual_test_balance.values

        metrics = self._compute_metrics(y_true_balance, y_pred_balance)

        self._plot_forecast_vs_actual(y_true_balance, y_pred_balance, plot_dir)
        self._plot_residuals(y_true_balance, y_pred_balance, plot_dir)
        self._plot_metrics_summary(metrics, plot_dir)

        return metrics

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ForecastMetrics:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        total_actual = np.sum(np.abs(y_true))
        wape = (
            float(np.sum(np.abs(y_true - y_pred)) / total_actual)
            if total_actual > 0
            else 0.0
        )

        nonzero = y_true != 0
        mape = (
            float(
                np.mean(
                    np.abs(
                        (y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]
                    )
                )
            )
            if nonzero.any()
            else 0.0
        )

        denom = np.abs(y_true) + np.abs(y_pred)
        smape = (
            float(np.mean(2 * np.abs(y_true - y_pred) / denom[denom > 0]))
            if (denom > 0).any()
            else 0.0
        )

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        bias = float(np.mean(y_pred - y_true))

        logger.info("MAE   : %.4f", mae)
        logger.info("RMSE  : %.4f", rmse)
        logger.info("WAPE  : %.4f", wape)
        logger.info("MAPE  : %.4f", mape)
        logger.info("sMAPE : %.4f", smape)
        logger.info("R²    : %.4f", r2)
        logger.info("Bias  : %.4f", bias)

        return ForecastMetrics(
            mae=mae,
            rmse=rmse,
            wape=wape,
            mape=mape,
            smape=smape,
            r2=r2,
            bias=bias,
        )

    def _plot_forecast_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        plot_dir: Path | None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(y_true, label="Actual Balance", color="#4C72B0", linewidth=1.5)
        ax.plot(
            y_pred,
            label="Forecast Balance",
            color="#C44E52",
            linewidth=1.5,
            linestyle="--",
        )
        ax.set_title("Forecast vs Actual", fontsize=13, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        self._save_or_show(fig, plot_dir, "forecast_vs_actual.png")

    def _plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        plot_dir: Path | None,
    ) -> None:
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(residuals, color="#55A868", linewidth=1)
        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].fill_between(
            range(len(residuals)), residuals, alpha=0.2, color="#55A868"
        )
        axes[0].set_title("Residuals over Time", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Actual − Forecast")
        axes[0].grid(alpha=0.3)

        axes[1].hist(
            residuals, bins=30, color="#8172B2", edgecolor="white", alpha=0.85
        )
        axes[1].axvline(
            float(np.mean(residuals)),
            color="#C44E52",
            linewidth=1.5,
            linestyle="--",
            label=f"Mean: {np.mean(residuals):.2f}",
        )
        axes[1].set_title(
            "Residual Distribution", fontsize=12, fontweight="bold"
        )
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.suptitle(
            "Residual Analysis", fontsize=13, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        self._save_or_show(fig, plot_dir, "residuals.png")

    def _plot_metrics_summary(
        self,
        metrics: ForecastMetrics,
        plot_dir: Path | None,
    ) -> None:
        labels = ["MAE", "RMSE", "WAPE", "MAPE", "sMAPE", "R²", "Bias"]
        values = [
            metrics.mae,
            metrics.rmse,
            metrics.wape,
            metrics.mape,
            metrics.smape,
            metrics.r2,
            metrics.bias,
        ]
        colors = [
            "#4C72B0",
            "#55A868",
            "#C44E52",
            "#8172B2",
            "#CCB974",
            "#64B5CD",
            "#AA6060",
        ]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(
            labels, values, color=colors, edgecolor="white", height=0.55
        )

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + abs(max(values, key=abs)) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
            )

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(
            "Forecasting Metrics Summary", fontsize=13, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        self._save_or_show(fig, plot_dir, "forecast_metrics_summary.png")

    @staticmethod
    def _save_or_show(
        fig: plt.Figure, plot_dir: Path | None, filename: str
    ) -> None:
        if plot_dir is not None:
            plot_dir.mkdir(parents=True, exist_ok=True)
            path = plot_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved plot: %s", path)
        else:
            plt.show()
        plt.close(fig)


def main() -> None:
    print("Loading forecasting dataset...")

    loader = ForecastingDataLoader()
    df = loader.load()

    if df.empty:
        raise ValueError("Forecasting dataset is empty.")

    preprocessor = ForecastPreprocessor()
    series_df = preprocessor.prepare_series(df)

    if series_df.empty:
        raise ValueError("Prepared forecasting series is empty.")

    series = series_df.set_index("ds")["y"]

    print(f"Prepared time series with {len(series)} observations.")

    model = TransactionForecaster(ml_settings)
    model.load_model()

    if model.model is None:
        raise FileNotFoundError(
            f"Model not found: {ml_settings.forecaster.model.path}"
        )

    split_idx = int(len(series) * (1 - ml_settings.data.test_size))
    train_series = series.iloc[:split_idx]
    test_series = series.iloc[split_idx:]

    # Calculate actual balances for evaluation
    last_train_balance = float(train_series.sum())
    actual_test_balance = last_train_balance + test_series.cumsum()

    print("Running evaluation...")

    evaluator = ForecastEvaluator()

    metrics = evaluator.evaluate(
        model=model,
        test_series=test_series,
        actual_test_balance=actual_test_balance,
        last_train_balance=last_train_balance,
        plot_dir=Path("artifacts/forecaster/evaluation"),
    )

    print("\nForecast Evaluation Results")
    print("-" * 40)
    print(f"MAE   : {metrics.mae:.4f}")
    print(f"RMSE  : {metrics.rmse:.4f}")
    print(f"WAPE  : {metrics.wape:.4f}")
    print(f"MAPE  : {metrics.mape:.4f}")
    print(f"sMAPE : {metrics.smape:.4f}")
    print(f"R²    : {metrics.r2:.4f}")
    print(f"Bias  : {metrics.bias:.4f}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
