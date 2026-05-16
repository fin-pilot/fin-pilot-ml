from typing import cast

import pandas as pd
from datasets import load_dataset

from ml.config import ml_settings


class CategorizerDataLoader:
    def __init__(self) -> None:
        self.dataset_name = ml_settings.categorizer.dataset.name

    def load(self) -> tuple[pd.Series, pd.Series]:
        print(f"Loading dataset: {self.dataset_name}")

        dataset = load_dataset(self.dataset_name)

        df = cast(pd.DataFrame, dataset["train"].to_pandas())
        df = df.dropna(subset=["transaction_description", "category"])

        df["description"] = (
            df["transaction_description"].astype(str).str.lower().str.strip()
        )

        print(f"Loaded {len(df)} cleaned samples.")

        return df["description"], df["category"]
