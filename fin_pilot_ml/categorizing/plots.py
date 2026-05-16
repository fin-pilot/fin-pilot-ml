from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


class CategorizingPlots:
    @staticmethod
    def confusion_matrix_plot(
        y_true,
        y_pred,
        labels,
        output_path: Path,
    ) -> None:

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        )

        plt.figure(
            figsize=(
                max(10, len(labels)),
                max(8, len(labels)),
            )
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )

        plt.title("Confusion Matrix")

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=200,
        )

        plt.close()

    @staticmethod
    def per_class_metrics_plot(
        report: dict,
        output_path: Path,
    ) -> None:

        classes = [
            key
            for key in report.keys()
            if key
            not in (
                "accuracy",
                "macro avg",
                "weighted avg",
            )
        ]

        df = pd.DataFrame(
            {
                cls: {
                    "precision": report[cls]["precision"],
                    "recall": report[cls]["recall"],
                    "f1-score": report[cls]["f1-score"],
                }
                for cls in classes
            }
        ).T

        df.plot(
            kind="bar",
            figsize=(14, 6),
        )

        plt.title("Per-class Metrics")

        plt.ylabel("Score")

        plt.ylim(0, 1)

        plt.xticks(rotation=45)

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=200,
        )

        plt.close()

    @staticmethod
    def metrics_summary_plot(
        metrics: dict,
        output_path: Path,
    ) -> None:

        df = pd.DataFrame(
            {
                "metric": list(metrics.keys()),
                "value": list(metrics.values()),
            }
        )

        plt.figure(figsize=(10, 5))

        sns.barplot(
            data=df,
            x="metric",
            y="value",
        )

        plt.ylim(0, 1)

        plt.title("Overall Metrics Summary")

        plt.xticks(rotation=30)

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=200,
        )

        plt.close()
