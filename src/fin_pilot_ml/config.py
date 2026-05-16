from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    test_size: float = 0.2
    random_state: int = 42


class ModelPathConfig(BaseModel):
    path: Path


class DatasetConfig(BaseModel):
    name: str


class TfidfConfig(BaseModel):
    analyzer: Literal["word", "char", "char_wb"] = "char_wb"
    ngram_range: tuple[int, int] = (2, 5)
    min_df: int = 5
    max_df: float = 0.5


class SVMConfig(BaseModel):
    class_weight: str | None = "balanced"
    max_iter: int = 2000
    random_state: int = 42


class CategorizerConfig(BaseModel):
    dataset: DatasetConfig
    model: ModelPathConfig
    tfidf: TfidfConfig
    svm: SVMConfig


class SarimaConfig(BaseModel):
    seasonal: bool = True
    freq: str = "W"
    seasonal_period: int = 4
    stepwise: bool = False
    trace: bool = True
    error_action: str = "ignore"
    suppress_warnings: bool = True
    information_criterion: Literal["aic", "bic", "hqic"] = "aic"
    max_p: int = 7
    max_q: int = 7
    max_d: int = 2
    max_P: int = 5
    max_Q: int = 5
    max_D: int = 1
    n_jobs: int = -1
    max_order: int | None = None


class ForecasterConfig(BaseModel):
    dataset: DatasetConfig
    model: ModelPathConfig
    sarima: SarimaConfig


class MLSettings(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    categorizer: CategorizerConfig
    forecaster: ForecasterConfig


_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_ml_settings() -> MLSettings:
    base = _load_yaml(_CONFIGS_DIR / "base.yaml")
    categorizer = _load_yaml(_CONFIGS_DIR / "categorizer.yaml")
    forecaster = _load_yaml(_CONFIGS_DIR / "forecaster.yaml")

    merged = {
        **base,
        **categorizer,
        **forecaster,
    }

    return MLSettings.model_validate(merged)


ml_settings: MLSettings = _load_ml_settings()
