from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ForecastMetrics:
    mae: float
    mse: float
    rmse: float
    mape: float
    smape: float
    wape: float
    r2: float
    mbe: float
    residual_mean: float
    residual_std: float
    plots_dir: Path
