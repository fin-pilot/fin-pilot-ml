import logging
from typing import cast

import pandas as pd
from datasets import load_dataset

from fin_pilot_ml.categorizing.config import (
    CategorizingConfig,
)

logger = logging.getLogger(__name__)


class CategorizingDataLoader:
    def __init__(
        self,
        config: CategorizingConfig,
    ) -> None:
        self.config = config

    def load(
        self,
    ) -> tuple[pd.Series, pd.Series]:

        logger.info(
            "Loading dataset: %s",
            self.config.dataset_name,
        )

        dataset = load_dataset(
            self.config.dataset_name,
        )

        df = cast(
            pd.DataFrame,
            dataset["train"].to_pandas(),
        )

        df = df.dropna(
            subset=[
                "transaction_description",
                "category",
            ]
        )

        df["description"] = (
            df["transaction_description"].astype(str).str.lower().str.strip()
        )

        logger.info(
            "Loaded %s samples.",
            len(df),
        )

        return (
            df["description"],
            df["category"],
        )
