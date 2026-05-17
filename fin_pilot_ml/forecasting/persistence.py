import joblib
from pathlib import Path
from typing import Any


class ModelPersistence:
    @staticmethod
    def save(model: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

    @staticmethod
    def load(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        return joblib.load(path)
