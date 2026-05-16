from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.config import MLSettings


class TransactionCategorizer:
    def __init__(self, config: MLSettings) -> None:
        self.config = config
        print("Initializing transaction categorizer.")
        self.pipeline = self._build_pipeline()

    def train(self, x_train: list[str], y_train: list[str]) -> None:
        print("Training transaction categorizer...")
        self.pipeline.fit(x_train, y_train)

    def predict(self, texts: list[str]) -> list[str]:
        cleaned = [str(t).lower().strip() for t in texts]
        return [str(p) for p in self.pipeline.predict(cleaned)]

    def save_model(self) -> None:
        model_path = self._model_path
        model_path.parent.patent.mkdir(parents=True, exist_ok=True)

        print(f"Saving model to {model_path}")

        joblib.dump(self.pipeline, model_path)

    def load_model(self) -> None:
        model_path = self._model_path

        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            return

        print(f"Loading model from {model_path}")

        self.pipeline = joblib.load(model_path)

    @property
    def _model_path(self) -> Path:
        return Path(self.config.categorizer.model.path)

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("tfidf", self._create_vectorizer()),
                ("clf", self._create_classifier()),
            ]
        )

    def _create_vectorizer(self) -> TfidfVectorizer:
        cfg = self.config.categorizer.tfidf

        return TfidfVectorizer(
            analyzer=cfg.analyzer,
            ngram_range=cfg.ngram_range,
            min_df=cfg.min_df,
            max_df=cfg.max_df,
        )

    def _create_classifier(self) -> LinearSVC:
        cfg = self.config.categorizer.svm

        return LinearSVC(
            class_weight=cfg.class_weight,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
        )
