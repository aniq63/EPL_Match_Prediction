from datetime import datetime, timedelta
import pandas as pd
from src.utils.logger import logging

# Helper function outside the class
def get_date_3_weeks_back():
    try:
        current_date = datetime.now()
        past_date = current_date - timedelta(weeks=5)
        return past_date
    except Exception as e:
        logging.error(f"Error calculating 3-weeks-back date: {e}")
        raise Exception(f"Error calculating 3-weeks-back date: {e}")


class DataSplitter:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def split(self):
        try:
            logging.info("Starting data split process...")
            # Check required column
            if "date" not in self.df.columns:
                logging.error("Column 'date' not found in dataframe")
                raise ValueError("Column 'date' not found in dataframe")

            # Convert to datetime
            self.df["date"] = pd.to_datetime(self.df["date"])

            # Get split date using helper function
            split_date = get_date_3_weeks_back()
            logging.info(f"Splitting data with split date: {split_date.date()}")

            # Perform split
            train_df = self.df[self.df["date"] < split_date].copy()
            test_df  = self.df[self.df["date"] >= split_date].copy()

            logging.info(f"Split complete. Train: {len(train_df)} rows, Test: {len(test_df)} rows")

            return train_df, test_df

        except Exception as e:
            logging.error(f"Data split failed: {e}")
            raise Exception(f"Data split failed: {e}")