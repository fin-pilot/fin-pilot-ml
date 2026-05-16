import pandas as pd


class ForecastPreprocessor:
    """
    Transforms raw transactions into a daily net flow time series.

    Income is positive, expenses are negative.
    Missing days are forward-filled with 0 to keep the series continuous.
    Outlier daily changes are capped using IQR.
    """

    def prepare_series(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["ds", "y"])

        df = df.copy()
        df = df.drop_duplicates()
        df = df.sort_values("date").reset_index(drop=True)

        # Signed net flow: income positive, expense negative
        df["ds"] = pd.to_datetime(df["date"]).dt.normalize()
        df["net"] = df.apply(
            lambda r: (
                r["amount"]
                if r["transaction_type"] == "income"
                else -r["amount"]
            ),
            axis=1,
        )

        # Daily net flow
        full_index = pd.date_range(
            start=df["ds"].min(),
            end=df["ds"].max(),
            freq="D",
        )
        daily_net = (
            df.groupby("ds")["net"].sum().reindex(full_index, fill_value=0.0)
        )

        # Filter outlier daily shocks
        daily_net = self._cap_outliers(daily_net)

        # Return the net flow (NOT the cumsum)
        flow_series = daily_net.reset_index()
        flow_series.columns = pd.Index(["ds", "y"])

        print(f"Flow series : {len(flow_series)} daily observations.")
        print(
            f"Range       : {flow_series['ds'].iloc[0].date()} → "
            f"{flow_series['ds'].iloc[-1].date()}"
        )

        return flow_series

    def _cap_outliers(self, series: pd.Series) -> pd.Series:
        """
        Caps extreme daily shocks using IQR on the series itself.
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr

        capped = series.clip(lower=lower, upper=upper)
        n_capped = int(((series < lower) | (series > upper)).sum())

        if n_capped:
            print(f"Capped {n_capped} outlier daily flows.")

        return capped
