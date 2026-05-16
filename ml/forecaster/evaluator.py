from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ForecastMetrics:
        logger.info("Evaluating forecasting model...")

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

    def evaluate_walk_forward(
        self,
        model_factory: Callable[[], object],
        series: pd.Series,
        initial_train_size: int = 52,
        step: int = 4,
        plot_dir: Path | None = None,
    ) -> ForecastMetrics:
        """
        Walk-forward (expanding window) validation.

        Args:
            model_factory: Zero-argument callable returning a fresh model instance.
            series: Full time series.
            initial_train_size: Observations in first training window.
            step: Steps to advance per iteration.
            plot_dir: Directory to save plots. None = show interactively.
        """
        logger.info("Starting walk-forward validation...")

        if len(series) <= initial_train_size:
            logger.warning("Series too short for walk-forward validation.")
            return ForecastMetrics(
                mae=0.0,
                rmse=0.0,
                wape=0.0,
                mape=0.0,
                smape=0.0,
                r2=0.0,
                bias=0.0,
            )

        all_true: list[float] = []
        all_pred: list[float] = []
        step_maes: list[float] = []
        step_indices: list[int] = []

        for i in range(initial_train_size, len(series), step):
            train_split = series.iloc[:i]
            test_split = series.iloc[i : i + step]

            try:
                model = model_factory()
                model.fit(train_split)
                forecast_df: pd.DataFrame = model.forecast(
                    steps=len(test_split)
                )

                y_true_step = test_split.values
                y_pred_step = forecast_df["predicted_y"].values

                all_true.extend(y_true_step.tolist())
                all_pred.extend(y_pred_step.tolist())
                step_maes.append(
                    float(mean_absolute_error(y_true_step, y_pred_step))
                )
                step_indices.append(i)

            except Exception as error:
                logger.error("Walk-forward failed at step %s: %s", i, error)

        if not all_true or not all_pred:
            logger.warning("No successful walk-forward predictions generated.")
            return ForecastMetrics(
                mae=0.0,
                rmse=0.0,
                wape=0.0,
                mape=0.0,
                smape=0.0,
                r2=0.0,
                bias=0.0,
            )

        y_true_arr = np.array(all_true)
        y_pred_arr = np.array(all_pred)

        metrics = self.evaluate(y_true_arr, y_pred_arr)

        self._plot_forecast_vs_actual(y_true_arr, y_pred_arr, plot_dir)
        self._plot_residuals(y_true_arr, y_pred_arr, plot_dir)
        self._plot_step_mae(step_indices, step_maes, plot_dir)
        self._plot_metrics_summary(metrics, plot_dir)

        return metrics

    def _plot_forecast_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        plot_dir: Path | None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(y_true, label="Actual", color="#4C72B0", linewidth=1.5)
        ax.plot(
            y_pred,
            label="Forecast",
            color="#C44E52",
            linewidth=1.5,
            linestyle="--",
        )

        ax.set_title(
            "Walk-forward: Forecast vs Actual", fontsize=13, fontweight="bold"
        )
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

    def _plot_step_mae(
        self,
        step_indices: list[int],
        step_maes: list[float],
        plot_dir: Path | None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            step_indices,
            step_maes,
            marker="o",
            color="#CCB974",
            linewidth=1.5,
            markersize=4,
        )
        ax.axhline(
            float(np.mean(step_maes)),
            color="#C44E52",
            linewidth=1,
            linestyle="--",
            label=f"Mean MAE: {np.mean(step_maes):.2f}",
        )
        ax.set_title(
            "MAE per Walk-forward Step", fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Training window size (observations)")
        ax.set_ylabel("MAE")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        self._save_or_show(fig, plot_dir, "step_mae.png")

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
