from pathlib import Path

import kagglehub
import pandas as pd

from ml.config import ml_settings

_EMPTY_DF = pd.DataFrame(
    columns=["transaction_date", "amount", "transaction_type"]
)


class ForecastingDataLoader:
    def __init__(self) -> None:
        self.dataset_name = ml_settings.forecaster.dataset.name

    def load(self) -> pd.DataFrame:
        print(f"Loading forecasting dataset: {self.dataset_name}")

        try:
            path = Path(kagglehub.dataset_download(self.dataset_name))
            csv_files = list(path.rglob("*.csv"))

            if not csv_files:
                print("No CSV files found in dataset.")
                return _EMPTY_DF.copy()

            df = pd.read_csv(csv_files[0])

            df.columns = (
                df.columns.str.strip().str.lower().str.replace(" ", "_")
            )

            formatted_df = pd.DataFrame(
                {
                    "transaction_date": pd.to_datetime(
                        df["date_time"], errors="coerce"
                    ).dt.tz_localize(None),
                    "amount": pd.to_numeric(
                        df["amount"], errors="coerce"
                    ).abs(),
                    "transaction_type": "expense",
                }
            )

            formatted_df = formatted_df.dropna(
                subset=["transaction_date", "amount"]
            )

            print(f"Loaded {len(formatted_df)} forecasting samples.")

            return formatted_df

        except Exception as error:
            print(f"Failed to load forecasting dataset: {error}")
            return _EMPTY_DF.copy()
