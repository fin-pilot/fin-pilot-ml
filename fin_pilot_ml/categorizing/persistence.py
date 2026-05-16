import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


class ModelPersistence:
    @staticmethod
    def save(
        model: Any,
        path: Path,
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Saving model to %s",
            path,
        )

        joblib.dump(model, path)

    @staticmethod
    def load(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        logger.info(
            "Loading model from %s",
            path,
        )

        return joblib.load(path)
