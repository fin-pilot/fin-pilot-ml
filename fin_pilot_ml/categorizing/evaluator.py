import json
import logging
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from fin_pilot_ml.categorizing.plots import CategorizingPlots
from fin_pilot_ml.categorizing.schemas import EvaluationMetrics

logger = logging.getLogger(__name__)


class CategorizingEvaluator:
    def evaluate(
        self, model, x_test, y_test, plots_dir: Path
    ) -> EvaluationMetrics:
        logger.info("Evaluating categorizer...")

        plots_dir.mkdir(parents=True, exist_ok=True)

        y_pred = model.predict(x_test)

        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        metrics_dict = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "balanced_accuracy": round(
                float(balanced_accuracy_score(y_test, y_pred)), 4
            ),
            "precision": round(
                float(
                    precision_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    )
                ),
                4,
            ),
            "recall": round(
                float(
                    recall_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    )
                ),
                4,
            ),
            "f1_score": round(
                float(
                    f1_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    )
                ),
                4,
            ),
            "mcc": round(float(matthews_corrcoef(y_test, y_pred)), 4),
            "kappa": round(float(cohen_kappa_score(y_test, y_pred)), 4),
        }

        logger.info("Evaluation metrics: %s", metrics_dict)

        labels = sorted(y_test.unique().tolist())

        CategorizingPlots.confusion_matrix_plot(
            y_test, y_pred, labels, plots_dir / "confusion_matrix.png"
        )
        CategorizingPlots.per_class_metrics_plot(
            report, plots_dir / "per_class_metrics.png"
        )
        CategorizingPlots.metrics_summary_plot(
            metrics_dict, plots_dir / "metrics_summary.png"
        )

        metrics_json = plots_dir / "metrics.json"

        with open(metrics_json, "w", encoding="utf-8") as file:
            json.dump(
                {**metrics_dict, "classification_report": (report)},
                file,
                indent=2,
            )

        return EvaluationMetrics(
            accuracy=metrics_dict["accuracy"],
            balanced_accuracy=metrics_dict["balanced_accuracy"],
            precision=metrics_dict["precision"],
            recall=metrics_dict["recall"],
            f1_score=metrics_dict["f1_score"],
            mcc=metrics_dict["mcc"],
            kappa=metrics_dict["kappa"],
            roc_auc=None,
            report=report,
            plots_dir=plots_dir,
        )
