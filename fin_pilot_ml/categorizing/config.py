from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CategorizingConfig:
    dataset_name: str = "mitulshah/transaction-categorization"

    test_size: float = 0.2
    random_state: int = 42

    artifacts_dir: Path = Path("artifacts/categorizing")
    models_dir: Path = Path("artifacts/categorizing/models")
    evaluation_dir: Path = Path("artifacts/categorizing/evaluation")
    model_path: Path = Path("artifacts/categorizing/models/model.pkl")

    analyzer: str = "word"
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    class_weight: str = "balanced"
    max_iter: int = 5000
