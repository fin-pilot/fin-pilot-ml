from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


class FinanceDatasetLoader:
    def __init__(
        self,
        dataset_name: str = "ramyapintchy/personal-finance-data",
    ) -> None:
        self.dataset_name = dataset_name

    def load(self) -> pd.DataFrame:
        api = KaggleApi()
        api.authenticate()

        kaggle_cache = (
            Path.home()
            / ".cache"
            / "kagglehub"
            / self.dataset_name.replace("/", "_")
        )

        kaggle_cache.mkdir(parents=True, exist_ok=True)

        api.dataset_download_files(
            self.dataset_name,
            path=kaggle_cache,
            unzip=True,
        )

        csv_files = list(kaggle_cache.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("No CSV files found in dataset.")

        file_path = csv_files[0]

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        return df