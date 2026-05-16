from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ForecastPlots:
    @staticmethod
    def actual_vs_forecast(
        actual: pd.Series,
        predicted: pd.Series,
        output_path: Path,
    ) -> None:

        plt.figure(figsize=(14, 6))

        plt.plot(
            actual.index,
            actual,
            label="Actual",
            linewidth=2,
        )

        plt.plot(
            predicted.index,
            predicted,
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

        plt.savefig(
            output_path,
            dpi=200,
        )

        plt.close()

    @staticmethod
    def residual_plot(
        residuals: pd.Series,
        output_path: Path,
    ) -> None:

        plt.figure(figsize=(14, 5))

        plt.plot(
            residuals.index,
            residuals,
        )

        plt.axhline(
            0,
            linestyle="--",
        )

        plt.title("Residual Errors")

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=200,
        )

        plt.close()

    @staticmethod
    def residual_distribution(
        residuals: pd.Series,
        output_path: Path,
    ) -> None:

        plt.figure(figsize=(10, 5))

        plt.hist(
            residuals,
            bins=20,
        )

        plt.title("Residual Distribution")

        plt.xlabel("Residual")

        plt.ylabel("Frequency")

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=200,
        )

        plt.close()

    @staticmethod
    def rolling_error_plot(
        actual: pd.Series,
        predicted: pd.Series,
        output_path: Path,
    ) -> None:

        rolling_error = np.abs(actual - predicted).rolling(window=4).mean()

        plt.figure(figsize=(14, 5))

        plt.plot(
            rolling_error.index,
            rolling_error,
        )

        plt.title("Rolling Forecast Error")

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=200,
        )

        plt.close()
