from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ForecastPlots:
    @staticmethod
    def train_test_split(
        train_data: pd.Series, test_data: pd.Series, output_path: Path
    ) -> None:
        ForecastPlots._ensure_parent(output_path)

        plt.figure(figsize=(14, 6))
        plt.plot(
            train_data.index, train_data.values, label="Train", linewidth=2
        )
        plt.plot(test_data.index, test_data.values, label="Test", linewidth=2)
        plt.title("Train/Test Split")
        plt.xlabel("Date")
        plt.ylabel("Expenses")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def actual_vs_forecast(
        actual: pd.Series, predicted: pd.Series, output_path: Path
    ) -> None:
        ForecastPlots._ensure_parent(output_path)

        plt.figure(figsize=(14, 6))
        plt.plot(actual.index, actual.values, label="Actual", linewidth=2)
        plt.plot(
            predicted.index,
            predicted.values,
            label="Forecast",
            linestyle="--",
            linewidth=2,
        )
        plt.title("Actual vs Forecast")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def residual_plot(residuals: pd.Series, output_path: Path) -> None:
        ForecastPlots._ensure_parent(output_path)

        plt.figure(figsize=(14, 5))
        plt.plot(residuals.index, residuals.values)
        plt.axhline(y=0, linestyle="--")
        plt.title("Residual Errors")
        plt.xlabel("Date")
        plt.ylabel("Residual")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def residual_distribution(residuals: pd.Series, output_path: Path) -> None:
        ForecastPlots._ensure_parent(output_path)

        plt.figure(figsize=(10, 5))
        plt.hist(residuals.values, bins=20)
        plt.title("Residual Distribution")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def rolling_error_plot(
        actual: pd.Series,
        predicted: pd.Series,
        output_path: Path,
        window: int = 4,
    ) -> None:
        ForecastPlots._ensure_parent(output_path)

        rolling_error = np.abs(actual - predicted).rolling(window=window).mean()

        plt.figure(figsize=(14, 5))
        plt.plot(rolling_error.index, rolling_error.values)
        plt.title(f"Rolling Forecast Error ({window})")
        plt.xlabel("Date")
        plt.ylabel("Rolling MAE")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def model_comparison(metrics: dict[str, object], output_path: Path) -> None:
        ForecastPlots._ensure_parent(output_path)

        model_names = list(metrics.keys())
        mae_values = [metrics[name].mae for name in model_names]
        rmse_values = [metrics[name].rmse for name in model_names]
        x = np.arange(len(model_names))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, mae_values, width, label="MAE")
        plt.bar(x + width / 2, rmse_values, width, label="RMSE")
        plt.xticks(x, model_names)
        plt.ylabel("Error")
        plt.title("Model Comparison")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _ensure_parent(output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
