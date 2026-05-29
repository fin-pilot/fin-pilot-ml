from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ForecastingConfig:
    seasonal_period: int = 4

    test_ratio: float = 0.2
    minimum_test_size: int = 4

    p_values: tuple[int, ...] = (0, 1, 2)
    d_values: tuple[int, ...] = (0, 1)
    q_values: tuple[int, ...] = (0, 1, 2)

    P_values: tuple[int, ...] = (0, 1)
    D_values: tuple[int, ...] = (0, 1)
    Q_values: tuple[int, ...] = (0, 1)

    artifacts_dir: Path = Path("artifacts/forecasting")
    models_dir: Path = Path("artifacts/forecasting/models")
    evaluation_dir: Path = Path("artifacts/forecasting/evaluation")

    sarima_model_file: str = "sarima.pkl"
    hw_model_file: str = "holt_winters.pkl"
