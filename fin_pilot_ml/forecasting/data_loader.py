import os

import kagglehub
import pandas as pd


class FinanceDatasetLoader:
    def __init__(
        self, dataset_name: str = "ramyapintchy/personal-finance-data"
    ) -> None:
        self.dataset_name = dataset_name

    def load(self) -> pd.DataFrame:
        path = kagglehub.dataset_download(self.dataset_name)

        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

        if not csv_files:
            raise FileNotFoundError("No CSV files found in dataset.")

        file_path = os.path.join(path, csv_files[0])

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        return df
