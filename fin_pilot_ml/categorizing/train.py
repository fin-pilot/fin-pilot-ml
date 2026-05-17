import logging
import time

from sklearn.model_selection import train_test_split

from fin_pilot_ml.categorizing.config import CategorizingConfig
from fin_pilot_ml.categorizing.data_loader import CategorizingDataLoader
from fin_pilot_ml.categorizing.evaluator import CategorizingEvaluator
from fin_pilot_ml.categorizing.model import TransactionCategorizer
from fin_pilot_ml.categorizing.persistence import ModelPersistence

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | %(levelname)s | " "%(name)s | %(message)s"),
)

logger = logging.getLogger(__name__)


class CategorizingTrainer:
    def __init__(self) -> None:
        self.config = CategorizingConfig()
        self.loader = CategorizingDataLoader(self.config)
        self.evaluator = CategorizingEvaluator()
        self.model = TransactionCategorizer(self.config)

    def run(self) -> None:
        start = time.perf_counter()

        logger.info("Starting categorizing pipeline.")

        x, y = self.loader.load()

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=(self.config.test_size),
            random_state=(self.config.random_state),
            stratify=y,
        )

        self.model.train(list(x_train), list(y_train))

        metrics = self.evaluator.evaluate(
            model=self.model.pipeline,
            x_test=x_test,
            y_test=y_test,
            plots_dir=(self.config.artifacts_dir / "evaluation"),
        )

        logger.info("Final metrics: %s", metrics)

        ModelPersistence.save(self.model.pipeline, self.config.model_path)

        elapsed = time.perf_counter() - start

        logger.info("Training complete in %.2fs", elapsed)


def main() -> None:
    trainer = CategorizingTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
