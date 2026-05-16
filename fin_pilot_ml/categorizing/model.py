import logging

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from fin_pilot_ml.categorizing.config import (
    CategorizingConfig,
)

logger = logging.getLogger(__name__)


class TransactionCategorizer:
    def __init__(
        self,
        config: CategorizingConfig,
    ) -> None:
        self.config = config

        self.pipeline = self._build_pipeline()

    def train(
        self,
        x_train: list[str],
        y_train: list[str],
    ) -> None:

        logger.info("Training categorizer model...")

        self.pipeline.fit(
            x_train,
            y_train,
        )

    def predict(
        self,
        texts: list[str],
    ) -> list[str]:

        cleaned = [str(text).lower().strip() for text in texts]

        predictions = self.pipeline.predict(
            cleaned,
        )

        return [str(pred) for pred in predictions]

    def _build_pipeline(
        self,
    ) -> Pipeline:

        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer=self.config.analyzer,
                        ngram_range=(self.config.ngram_range),
                        min_df=self.config.min_df,
                        max_df=self.config.max_df,
                    ),
                ),
                (
                    "clf",
                    LinearSVC(
                        class_weight=(self.config.class_weight),
                        max_iter=(self.config.max_iter),
                        random_state=(self.config.random_state),
                    ),
                ),
            ]
        )
