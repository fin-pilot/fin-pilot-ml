from fin_pilot_ml.forecaster.data_loader import ForecastingDataLoader
from fin_pilot_ml.forecaster.evaluator import ForecastEvaluator
from fin_pilot_ml.forecaster.model import TransactionForecaster
from fin_pilot_ml.forecaster.preprocessing import ForecastPreprocessor
from fin_pilot_ml.config import ml_settings


class ForecastTrainer:
    def __init__(self) -> None:
        self.data_loader = ForecastingDataLoader()
        self.preprocessor = ForecastPreprocessor()
        self.evaluator = ForecastEvaluator()
        self.model = TransactionForecaster(ml_settings)

    def run(self) -> None:
        print("Starting forecasting pipeline.")

        df = self.data_loader.load()

        if df.empty:
            raise ValueError("Forecasting dataset is empty.")

        series_df = self.preprocessor.prepare_series(
            df=df,
            target="expense",
        )

        if series_df.empty:
            raise ValueError("Prepared forecasting series is empty.")

        series = series_df.set_index("ds")["y"]

        print(f"Time series: {len(series)} observations.")

        print(f"Date range : " f"{series.index.min()} → {series.index.max()}")

        print(
            f"Mean={float(series.mean()):.2f}  "
            f"Std={float(series.std()):.2f}"
        )

        self.model.check_stationarity(series)

        metrics = self.evaluator.evaluate_walk_forward(
            model_factory=lambda: TransactionForecaster(ml_settings),
            series=series,
            initial_train_size=max(
                14,
                int(len(series) * 0.6),
            ),
            step=max(
                1,
                int(len(series) * 0.1),
            ),
        )

        print(f"Walk-forward MAE : {metrics.mae:.4f}")
        print(f"Walk-forward RMSE: {metrics.rmse:.4f}")
        print(f"Walk-forward WAPE: {metrics.wape:.4f}")

        if metrics.wape > 1.0:
            print(
                "Model quality is poor (WAPE > 1.0). "
                "Forecasts may be unreliable."
            )

        print("Training forecasting model on full dataset...")

        self.model.fit(series)

        self.model.save_model()

        print("Forecasting training completed successfully.")


def main() -> None:
    trainer = ForecastTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
