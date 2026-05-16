from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class EvaluationMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_score: float
    mcc: float
    kappa: float
    roc_auc: float | None
    report: dict
    plots_dir: Path
