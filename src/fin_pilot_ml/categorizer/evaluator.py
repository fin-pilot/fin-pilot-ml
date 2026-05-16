from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


@dataclass(slots=True)
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    report: dict
    confusion_matrix: np.ndarray | None


class CategorizerEvaluator:
    def evaluate(
        self,
        model: Pipeline,
        x_test: pd.Series,
        y_test: pd.Series,
        show_confusion_matrix: bool = False,
    ) -> EvaluationMetrics:
        print("Evaluating categorizer model...")

        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        report: dict = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        print(
            "\nClassification Report:\n"
            f"{classification_report(y_test, y_pred, zero_division=0)}"
        )

        cm = None
        if show_confusion_matrix:
            cm = confusion_matrix(y_test, y_pred)
            print(f"Confusion matrix:\n{cm}")

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            report=report,
            confusion_matrix=cm,
        )
