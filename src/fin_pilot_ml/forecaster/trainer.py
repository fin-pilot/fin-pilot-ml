import logging

from fin_pilot_ml.forecaster.data_loader import ForecastingDataLoader
from fin_pilot_ml.forecaster.evaluator import ForecastEvaluator
from fin_pilot_ml.forecaster.model import TransactionForecaster
from fin_pilot_ml.forecaster.preprocessing import ForecastPreprocessor
from fin_pilot_ml.config import ml_settings

logger = logging.getLogger(__name__)


class ForecastTrainer:
    def __init__(self) -> None:
        self.data_loader = ForecastingDataLoader()
        self.preprocessor = ForecastPreprocessor()
        self.evaluator = ForecastEvaluator()
        self.model = TransactionForecaster(ml_settings)

    def run(self) -> None:
        logger.info("Starting forecasting pipeline.")

        df = self.data_loader.load()
        if df.empty:
            raise ValueError("Forecasting dataset is empty.")

        series_df = self.preprocessor.prepare_series(df=df, target="expense")
        if series_df.empty:
            raise ValueError("Prepared forecasting series is empty.")

        series = series_df.set_index("ds")["y"]

        logger.info("Time series: %s observations.", len(series))
        logger.info(
            "Date range : %s → %s", series.index.min(), series.index.max()
        )
        logger.info(
            "Mean=%.2f  Std=%.2f", float(series.mean()), float(series.std())
        )

        self.model.check_stationarity(series)

        metrics = self.evaluator.evaluate_walk_forward(
            model_factory=lambda: TransactionForecaster(ml_settings),
            series=series,
            initial_train_size=max(14, int(len(series) * 0.6)),
            step=max(1, int(len(series) * 0.1)),
        )

        logger.info("Walk-forward MAE : %.4f", metrics.mae)
        logger.info("Walk-forward RMSE: %.4f", metrics.rmse)
        logger.info("Walk-forward WAPE: %.4f", metrics.wape)

        if metrics.wape > 1.0:
            logger.warning(
                "Model quality is poor (WAPE > 1.0). "
                "Forecasts may be unreliable."
            )

        logger.info("Training forecasting model on full dataset...")
        self.model.fit(series)

        self.model.save_model()
        logger.info("Forecasting training completed successfully.")


def main() -> None:
    trainer = ForecastTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
