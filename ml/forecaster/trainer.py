import pandas as pd

from ml.forecaster.data_loader import ForecastingDataLoader
from ml.forecaster.evaluator import ForecastEvaluator
from ml.forecaster.model import TransactionForecaster
from ml.forecaster.preprocessing import ForecastPreprocessor
from ml.config import ml_settings


class ForecastTrainer:
    def __init__(self) -> None:
        self.config = ml_settings
        self.data_loader = ForecastingDataLoader()
        self.preprocessor = ForecastPreprocessor()
        self.evaluator = ForecastEvaluator()
        self.model = TransactionForecaster(self.config)

    def run(self) -> None:
        print("Starting balance forecasting pipeline.")

        df = self.data_loader.load()

        if df.empty:
            raise ValueError("Forecasting dataset is empty.")

        series_df = self.preprocessor.prepare_series(df)

        if series_df.empty:
            raise ValueError("Prepared flow series is empty.")

        series = series_df.set_index("ds")["y"]

        print(f"Date range : {series.index.min()} → {series.index.max()}")
        print(
            f"Flow Mean={float(series.mean()):.2f}  Std={float(series.std()):.2f}"
        )

        train_series, test_series = self._split_series(series)

        # Reconstruct actual balance to pass to evaluator
        last_train_balance = float(train_series.sum())
        actual_test_balance = last_train_balance + test_series.cumsum()

        self.model.train(train_series)

        metrics = self.evaluator.evaluate(
            model=self.model,
            test_series=test_series,
            actual_test_balance=actual_test_balance,
            last_train_balance=last_train_balance,
        )

        print(f"Final MAE  : {metrics.mae:.4f}")
        print(f"Final RMSE : {metrics.rmse:.4f}")
        print(f"Final WAPE : {metrics.wape:.4f}")

        if metrics.wape > 1.0:
            print(
                "Model quality is poor (WAPE > 1.0). "
                "Balance forecasts may be unreliable."
            )

        self.model.save_model()

        print("Balance forecasting training completed.")

    def _split_series(
        self,
        series: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        split_idx = int(len(series) * (1 - self.config.data.test_size))

        return series.iloc[:split_idx], series.iloc[split_idx:]


def main() -> None:
    trainer = ForecastTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
