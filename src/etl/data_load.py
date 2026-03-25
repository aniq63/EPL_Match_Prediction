import pandas as pd
import sys
from src.utils.logger import logging
from sqlalchemy import create_engine

from config.constants import TABLE_NAME, ETL_REQUIRED_COLUMNS
from src.utils.setting import get_settings


class DataLoad:
    """
    Loads transformed EPL match data into Supabase (PostgreSQL) data warehouse.
    Validates required columns before insertion and handles the database connection.
    """
    
    REQUIRED_COLUMNS = ETL_REQUIRED_COLUMNS
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataLoad with transformed DataFrame.
        
        Args:
            df (pd.DataFrame): Transformed data from DataTransformation
        """
        self.df = df

    def validate_columns(self) -> bool:
        """
        Validate that all required columns exist in the DataFrame.
        
        Returns:
            bool: True if all columns are present
            
        Raises:
            MyException: If required columns are missing
        """
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns for data load: {missing_columns}"
            logging.error(error_msg)
            raise Exception(error_msg)
        
        logging.info(f"Column validation passed. All {len(self.REQUIRED_COLUMNS)} required columns present.")
        return True

    def load_data_Supabase(self) -> bool:
        """
        Load transformed data into Supabase (PostgreSQL).
        Validates all required columns before insertion.
        Uses if_exists='replace' to refresh the complete dataset.
        
        Returns:
            bool: True if load was successful
            
        Raises:
            MyException: If validation or loading fails
        """
        try:
            # Validate columns before proceeding
            self.validate_columns()
            
            settings = get_settings()
            
            logging.info("Preparing database connection for Supabase...")
            # Convert async database URLs to sync for pandas SQLAlchemy compatibility
            db_url = settings.database_url
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            elif db_url.startswith("postgresql+asyncpg://"):
                db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)
            
            logging.info("Creating SQLAlchemy synchronous engine...")
            engine = create_engine(db_url)
            
            logging.info(f"Pushing {len(self.df)} records into '{TABLE_NAME}' table...")
            logging.info("Note: Using if_exists='replace' — existing data will be completely replaced")
            
            # Use if_exists='replace' to refresh dataset with extracted seasons
            self.df.to_sql(
                name=TABLE_NAME.lower(), 
                con=engine, 
                if_exists='replace', 
                index=False
            )
            
            logging.info(f"Successfully loaded {len(self.df)} records into Supabase table: {TABLE_NAME.lower()}")
            return True
            
        except Exception as e:
            logging.error(f"Error occurred during Supabase data load: {str(e)}")
            raise
