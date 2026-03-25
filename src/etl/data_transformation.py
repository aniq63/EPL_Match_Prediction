import pandas as pd
import sys
from src.utils.logger import logging
from config.constants import ETL_COLUMNS_TO_DROP

import warnings
warnings.filterwarnings('ignore')


class DataTransformation:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def drop_useless_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        existing = [c for c in ETL_COLUMNS_TO_DROP if c in df.columns]
        df = df.drop(columns=existing)
        logging.info(f"Dropped {len(existing)} columns: {existing}")
        return df

    def transform_pl_data(self):
        try:
            logging.info("Starting data transformation")
            df_pl = self.drop_useless_columns(self.df.copy())
            logging.info(f"Data transformation completed. Total records: {len(df_pl)}")
            return df_pl
        except Exception as e:
            logging.error(f"Error occurred during data transformation: {str(e)}")
            raise