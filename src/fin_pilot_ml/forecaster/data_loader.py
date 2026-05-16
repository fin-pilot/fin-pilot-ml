import logging
from pathlib import Path

import kagglehub
import pandas as pd

from fin_pilot_ml.config import ml_settings

logger = logging.getLogger(__name__)

_EMPTY_DF = pd.DataFrame(
    columns=["transaction_date", "amount", "transaction_type"]
)


class ForecastingDataLoader:
    def __init__(self) -> None:
        self.dataset_name = ml_settings.forecaster.dataset.name

    def load(self) -> pd.DataFrame:
        logger.info("Loading forecasting dataset: %s", self.dataset_name)

        try:
            path = Path(kagglehub.dataset_download(self.dataset_name))
            csv_files = list(path.rglob("*.csv"))

            if not csv_files:
                logger.error("No CSV files found in dataset.")
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

            logger.info("Loaded %s forecasting samples.", len(formatted_df))

            return formatted_df

        except Exception as error:
            logger.error("Failed to load forecasting dataset: %s", error)
            return _EMPTY_DF.copy()
