from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

from ml.categorizer.data_loader import CategorizerDataLoader
from ml.config import ml_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mcc: float
    kappa: float
    roc_auc: float
    report: dict
    confusion_matrix: np.ndarray | None = field(default=None)


class CategorizerEvaluator:
    def evaluate(
        self,
        model: Pipeline,
        x_test: pd.Series,
        y_test: pd.Series,
        show_confusion_matrix: bool = True,
        plot_dir: Path | None = None,
    ) -> EvaluationMetrics:
        logger.info("Evaluating categorizer model...")

        y_pred = model.predict(x_test)
        classes = sorted(y_test.unique().tolist())

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        try:
            lb = LabelBinarizer().fit(y_test)
            y_prob = model.predict_proba(x_test)
            roc_auc = roc_auc_score(
                lb.transform(y_test), y_prob, multi_class="ovr", average="macro"
            )
        except Exception:
            logger.warning(
                "ROC AUC unavailable (model may not support predict_proba)."
            )
            roc_auc = float("nan")

        report: dict = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        logger.info("Accuracy  : %.4f", accuracy)
        logger.info("Precision : %.4f", precision)
        logger.info("Recall    : %.4f", recall)
        logger.info("F1 Score  : %.4f", f1)
        logger.info("MCC       : %.4f", mcc)
        logger.info("Kappa     : %.4f", kappa)
        logger.info("ROC AUC   : %.4f", roc_auc)
        logger.info(
            "\nClassification Report:\n%s",
            classification_report(y_test, y_pred, zero_division=0),
        )

        cm = None
        if show_confusion_matrix:
            cm = confusion_matrix(y_test, y_pred, labels=classes)

        self._plot_confusion_matrix(cm, classes, plot_dir)
        self._plot_per_class_f1(report, classes, plot_dir)
        self._plot_metrics_summary(
            accuracy, precision, recall, f1, mcc, kappa, plot_dir
        )

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mcc=mcc,
            kappa=kappa,
            roc_auc=roc_auc,
            report=report,
            confusion_matrix=cm,
        )

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray | None,
        classes: list[str],
        plot_dir: Path | None,
    ) -> None:
        if cm is None:
            return

        fig, ax = plt.subplots(
            figsize=(max(8, len(classes)), max(6, len(classes) - 1))
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        self._save_or_show(fig, plot_dir, "confusion_matrix.png")

    def _plot_per_class_f1(
        self,
        report: dict,
        classes: list[str],
        plot_dir: Path | None,
    ) -> None:
        metrics = {
            cls: {
                "Precision": report[cls]["precision"],
                "Recall": report[cls]["recall"],
                "F1": report[cls]["f1-score"],
            }
            for cls in classes
            if cls in report
        }
        if not metrics:
            return

        df = pd.DataFrame(metrics).T
        x = np.arange(len(df))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.8), 5))
        ax.bar(
            x - width,
            df["Precision"],
            width,
            label="Precision",
            color="#4C72B0",
        )
        ax.bar(x, df["Recall"], width, label="Recall", color="#55A868")
        ax.bar(x + width, df["F1"], width, label="F1", color="#C44E52")
        ax.set_title("Per-class Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._save_or_show(fig, plot_dir, "per_class_metrics.png")

    def _plot_metrics_summary(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        mcc: float,
        kappa: float,
        plot_dir: Path | None,
    ) -> None:
        labels = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "MCC (norm)",
            "Kappa (norm)",
        ]
        values = [
            accuracy,
            precision,
            recall,
            f1,
            (mcc + 1) / 2,
            (kappa + 1) / 2,
        ]
        colors = [
            "#4C72B0",
            "#55A868",
            "#C44E52",
            "#8172B2",
            "#CCB974",
            "#64B5CD",
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(
            labels, values, color=colors, edgecolor="white", height=0.55
        )
        for bar, val in zip(bars, values):
            ax.text(
                min(val + 0.01, 0.97),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=9,
            )
        ax.set_xlim(0, 1.1)
        ax.set_title(
            "Overall Evaluation Summary", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Score")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        self._save_or_show(fig, plot_dir, "metrics_summary.png")

    @staticmethod
    def _save_or_show(
        fig: plt.Figure, plot_dir: Path | None, filename: str
    ) -> None:
        if plot_dir is not None:
            plot_dir.mkdir(parents=True, exist_ok=True)
            path = plot_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved plot: %s", path)
        else:
            plt.show()
        plt.close(fig)


def main() -> None:
    print("Loading dataset...")

    loader = CategorizerDataLoader()

    x, y = loader.load()

    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=ml_settings.data.test_size,
        random_state=ml_settings.data.random_state,
        stratify=y,
    )

    model_path = Path(ml_settings.categorizer.model.path)

    print(f"Loading model from {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    print("Evaluating trained model...")

    evaluator = CategorizerEvaluator()

    evaluator.evaluate(
        model=model,
        x_test=x_test,
        y_test=y_test,
        show_confusion_matrix=True,
        plot_dir=Path("artifacts/evaluation"),
    )

    print("Evaluation completed.")


if __name__ == "__main__":
    main()
