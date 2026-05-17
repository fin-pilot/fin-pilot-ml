import pandas as pd


class ExpensePreprocessor:
    DATE_COLUMN = "Date"
    AMOUNT_COLUMN = "Amount"

    def preprocess(self, df: pd.DataFrame) -> pd.Series:
        type_column = (
            "Transaction Type" if "Transaction Type" in df.columns else "Type"
        )

        df[self.DATE_COLUMN] = pd.to_datetime(
            df[self.DATE_COLUMN], format="mixed", errors="coerce"
        )
        df = df.dropna(subset=[self.DATE_COLUMN])

        expenses_df = df[
            df[type_column]
            .astype(str)
            .str.contains("Debit|Expense", case=False, na=False)
        ].copy()
        expenses_df.set_index(self.DATE_COLUMN, inplace=True)
        weekly_expenses = (
            expenses_df[self.AMOUNT_COLUMN].resample("W").sum().fillna(0)
        )

        non_zero = weekly_expenses[weekly_expenses > 0]
        if non_zero.empty:
            raise ValueError("No expense data found.")

        start_date = non_zero.index[0]

        return weekly_expenses.loc[start_date:]
