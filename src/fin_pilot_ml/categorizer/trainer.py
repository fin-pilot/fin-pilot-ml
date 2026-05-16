import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from fin_pilot_ml.categorizer.data_loader import CategorizerDataLoader
from fin_pilot_ml.categorizer.evaluator import CategorizerEvaluator
from fin_pilot_ml.categorizer.model import TransactionCategorizer
from fin_pilot_ml.config import ml_settings

logger = logging.getLogger(__name__)


class CategorizerTrainer:
    def __init__(self) -> None:
        self.config = ml_settings
        self.data_loader = CategorizerDataLoader()
        self.evaluator = CategorizerEvaluator()
        self.model = TransactionCategorizer(self.config)

    def run(self) -> None:
        logger.info("Starting categorizer training pipeline.")

        x, y = self.data_loader.load()
        x_train, x_test, y_train, y_test = self._split_data(x, y)

        self.model.train(list(x_train), list(y_train))

        metrics = self.evaluator.evaluate(
            model=self.model.pipeline,
            x_test=x_test,
            y_test=y_test,
        )

        logger.info("Final F1   : %.4f", metrics.f1_score)
        logger.info("Final Acc  : %.4f", metrics.accuracy)

        self.model.save_model()

        logger.info("Categorizer training completed.")

    def _split_data(
        self,
        x: pd.Series,
        y: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        return train_test_split(
            x,
            y,
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state,
            stratify=y,
        )


def main() -> None:
    trainer = CategorizerTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
