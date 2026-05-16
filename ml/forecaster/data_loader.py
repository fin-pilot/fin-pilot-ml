from pathlib import Path

import kagglehub
import pandas as pd

from ml.config import ml_settings

_EMPTY_DF = pd.DataFrame(columns=["date", "amount", "transaction_type"])


class ForecastingDataLoader:
    def __init__(self) -> None:
        self.dataset_name = ml_settings.forecaster.dataset.name

    def load(self) -> pd.DataFrame:
        print(f"Loading dataset: {self.dataset_name}")

        try:
            path = Path(kagglehub.dataset_download(self.dataset_name))

            expenses = self._load_file(path, "Expenses_clean.csv", "expense")
            income = self._load_file(path, "Income_clean.csv", "income")

            df = pd.concat([expenses, income], ignore_index=True)
            df = df.sort_values("date").reset_index(drop=True)

            print(
                f"Loaded {len(df)} transactions "
                f"({len(expenses)} expenses, {len(income)} income)."
            )

            return df

        except Exception as error:
            print(f"Failed to load dataset: {error}")
            return _EMPTY_DF.copy()

    def _load_file(
        self,
        path: Path,
        filename: str,
        transaction_type: str,
    ) -> pd.DataFrame:
        csv_path = next(path.rglob(filename), None)

        if csv_path is None:
            print(f"File not found: {filename}")
            return _EMPTY_DF.copy()

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        return pd.DataFrame(
            {
                "date": pd.to_datetime(
                    df["date_time"], errors="coerce"
                ).dt.tz_localize(None),
                "amount": pd.to_numeric(df["amount"], errors="coerce").abs(),
                "transaction_type": transaction_type,
            }
        ).dropna(subset=["date", "amount"])
