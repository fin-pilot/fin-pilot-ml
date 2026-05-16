from dataclasses import asdict
import json
import logging
import time
import warnings

from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
)

from fin_pilot_ml.forecasting.config import (
    ForecastingConfig,
)
from fin_pilot_ml.forecasting.data_loader import (
    FinanceDatasetLoader,
)
from fin_pilot_ml.forecasting.evaluator import (
    ForecastEvaluator,
)
from fin_pilot_ml.forecasting.models import (
    ForecastModels,
)
from fin_pilot_ml.forecasting.persistence import (
    ModelPersistence,
)
from fin_pilot_ml.forecasting.plots import (
    ForecastPlots,
)
from fin_pilot_ml.forecasting.preprocessing import (
    ExpensePreprocessor,
)
from fin_pilot_ml.forecasting.tuner import (
    SarimaTuner,
)

# =========================================================
# WARNING FILTERS
# =========================================================
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
)

warnings.filterwarnings(
    "ignore",
    message=("Non-invertible starting seasonal " "moving average"),
)

warnings.filterwarnings(
    "ignore",
    message=("Non-stationary starting seasonal " "autoregressive"),
)

warnings.filterwarnings(
    "ignore",
    message=("Too few observations to estimate " "starting parameters"),
)

warnings.filterwarnings(
    "ignore",
    message=("Non-invertible starting MA " "parameters found"),
)

warnings.filterwarnings(
    "ignore",
    message=("Non-stationary starting " "autoregressive parameters found"),
)

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | %(levelname)s | " "%(name)s | %(message)s"),
)

logger = logging.getLogger(__name__)


def main() -> None:
    start_time = time.perf_counter()

    logger.info("Starting forecasting pipeline.")

    config = ForecastingConfig()

    evaluation_dir = config.artifacts_dir / "evaluation"

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # LOAD DATA
    # =====================================================
    logger.info("Loading dataset...")

    loader = FinanceDatasetLoader()

    df = loader.load()

    logger.info(
        "Loaded %s rows.",
        len(df),
    )

    # =====================================================
    # PREPROCESS
    # =====================================================
    logger.info("Preprocessing data...")

    preprocessor = ExpensePreprocessor()

    series = preprocessor.preprocess(df)

    logger.info(
        "Prepared %s observations.",
        len(series),
    )

    if len(series) < 24:
        raise ValueError("Not enough observations " "for forecasting.")

    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================
    test_size = max(
        config.minimum_test_size,
        int(len(series) * config.test_ratio),
    )

    train_data = series.iloc[:-test_size]

    test_data = series.iloc[-test_size:]

    logger.info(
        "Train size=%s | Test size=%s",
        len(train_data),
        len(test_data),
    )

    ForecastPlots.train_test_split(
        train_data=train_data,
        test_data=test_data,
        output_path=(evaluation_dir / "train_test_split.png"),
    )

    # =====================================================
    # SARIMA
    # =====================================================
    use_sarima = len(train_data) >= 52

    sarima_model = None

    sarima_forecast = None

    if use_sarima:
        logger.info("Running SARIMA tuning...")

        tuner = SarimaTuner(config)

        (
            best_order,
            best_seasonal_order,
        ) = tuner.tune(
            train_data,
        )

        logger.info(
            "Best SARIMA params | " "order=%s seasonal=%s",
            best_order,
            best_seasonal_order,
        )

        logger.info("Training SARIMA model...")

        sarima_model = ForecastModels.train_sarima(
            train_data=train_data,
            order=best_order,
            seasonal_order=(best_seasonal_order),
        )

        sarima_forecast = sarima_model.forecast(
            steps=len(test_data),
        )

        sarima_forecast.index = test_data.index

    else:
        logger.warning("Skipping SARIMA due to " "insufficient observations.")

    # =====================================================
    # HOLT-WINTERS
    # =====================================================
    logger.info("Training Holt-Winters model...")

    hw_model = ForecastModels.train_holt_winters(
        train_data=train_data,
        seasonal_period=(config.seasonal_period),
    )

    hw_forecast = hw_model.forecast(
        steps=len(test_data),
    )

    hw_forecast.index = test_data.index

    # =====================================================
    # EVALUATION
    # =====================================================
    logger.info("Evaluating models...")

    metrics = {}

    if sarima_forecast is not None:
        metrics["SARIMA"] = ForecastEvaluator.evaluate(
            actual=test_data,
            predicted=sarima_forecast,
            plots_dir=evaluation_dir,
            model_name="sarima",
        )

    metrics["Holt-Winters"] = ForecastEvaluator.evaluate(
        actual=test_data,
        predicted=hw_forecast,
        plots_dir=evaluation_dir,
        model_name="holt_winters",
    )

    for (
        model_name,
        model_metrics,
    ) in metrics.items():

        logger.info(
            "%s metrics: %s",
            model_name,
            model_metrics,
        )

    ForecastPlots.model_comparison(
        metrics=metrics,
        output_path=(evaluation_dir / "model_comparison.png"),
    )

    # =====================================================
    # BEST MODEL SELECTION
    # =====================================================
    best_model_name = min(
        metrics,
        key=lambda x: metrics[x].rmse,
    )

    logger.info(
        "Best model selected: %s",
        best_model_name,
    )

    # =====================================================
    # SAVE MODELS
    # =====================================================
    logger.info("Saving models...")

    if sarima_model is not None:
        ModelPersistence.save(
            sarima_model,
            config.models_dir / config.sarima_model_file,
        )

    ModelPersistence.save(
        hw_model,
        config.models_dir / config.hw_model_file,
    )

    # =====================================================
    # SAVE SUMMARY
    # =====================================================
    summary = {}

    for model_name, model_metrics in metrics.items():
        data = asdict(model_metrics)

        if "plots_dir" in data:
            data["plots_dir"] = str(data["plots_dir"])

        summary[model_name] = data

    with open(
        evaluation_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    metadata = {
        "observations": len(series),
        "train_size": len(train_data),
        "test_size": len(test_data),
        "seasonal_period": (config.seasonal_period),
        "best_model": best_model_name,
    }

    with open(
        evaluation_dir / "pipeline_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # =====================================================
    # COMPLETE
    # =====================================================
    elapsed = time.perf_counter() - start_time

    logger.info(
        "Training pipeline complete " "in %.2f seconds.",
        elapsed,
    )


if __name__ == "__main__":
    try:
        main()

    except (
        ValueError,
        RuntimeError,
        OSError,
    ):
        logger.exception("Forecasting pipeline failed.")

        raise
