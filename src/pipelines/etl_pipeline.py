import pandas as pd
import sys
from datetime import datetime
from src.utils.logger import logging
from src.etl.data_extraction import DataExtraction
from src.etl.data_transformation import DataTransformation
from src.etl.data_load import DataLoad
import warnings

warnings.filterwarnings('ignore')


def get_seasons(base_year: int = 2023) -> list:
    """
    Returns all Understat season strings from base_year to current season.
    Understat format: '2023/2024', '2024/2025', etc.
    New season starts in August.
    
    Args:
        base_year (int): Starting year for season extraction (default: 2023)
        
    Returns:
        list: List of seasons in 'YYYY/YYYY' format
    """
    now = datetime.now()
    current_start = now.year if now.month >= 8 else now.year - 1
    return [f"{y}/{y+1}" for y in range(base_year, current_start + 1)]


class ETLPipeline:
    """
    This class orchestrates the entire ETL (Extract, Transform, Load) pipeline
    for Premier League match data processing.
    """

    def __init__(self, seasons: list):
        """
        Initialize the ETL Pipeline with seasons

        Args:
            seasons (list): List of seasons to extract data for
        """
        try:
            logging.info(f"Initializing ETL Pipeline with seasons: {seasons}")
            self.seasons = seasons
            self.extracted_df = None
            self.transformed_df = None
        except Exception as e:
            logging.error(f"Error occurred during ETL Pipeline initialization: {str(e)}")
            raise

    def extract_data(self):
        """
        Execute the data extraction step

        Returns:
            pd.DataFrame: Extracted data from Premier League
        """
        try:
            logging.info("Starting data extraction step...")
            data_extraction = DataExtraction(seasons=self.seasons)
            self.extracted_df = data_extraction.extract_pl_data()
            logging.info("Data extraction step completed successfully")
            return self.extracted_df
        except Exception as e:
            logging.error(f"Error occurred during data extraction step: {str(e)}")
            raise

    def transform_data(self):
        """
        Execute the data transformation step

        Returns:
            pd.DataFrame: Transformed data
        """
        try:
            if self.extracted_df is None:
                logging.warning("Extracted data is None. Running extraction first...")
                self.extract_data()

            logging.info("Starting data transformation step...")
            data_transformation = DataTransformation(df=self.extracted_df)
            self.transformed_df = data_transformation.transform_pl_data()
            logging.info("Data transformation step completed successfully")
            return self.transformed_df
        except Exception as e:
            logging.error(f"Error occurred during data transformation step: {str(e)}")
            raise

    def load_data(self):
        """
        Execute the data load step

        Returns:
            bool: True if loading was successful
        """
        try:
            if self.transformed_df is None:
                logging.warning("Transformed data is None. Running transformation first...")
                self.transform_data()

            logging.info("Starting data load step...")
            data_load = DataLoad(df=self.transformed_df)
            data_load.load_data_Supabase()
            logging.info("Data load step completed successfully")
            return True
        except Exception as e:
            logging.error(f"Error occurred during data load step: {str(e)}")
            raise

    def run(self):
        """
        Execute the complete ETL pipeline (Extract -> Transform -> Load)

        Returns:
            dict: Pipeline execution status and results
        """
        try:
            logging.info("=" * 60)
            logging.info("Starting ETL Pipeline execution...")
            logging.info("=" * 60)

            # Step 1: Extract
            logging.info("STEP 1: Data Extraction")
            logging.info("-" * 60)
            self.extract_data()

            # Step 2: Transform
            logging.info("STEP 2: Data Transformation")
            logging.info("-" * 60)
            self.transform_data()

            # Step 3: Load
            logging.info("STEP 3: Data Load")
            logging.info("-" * 60)
            self.load_data()

            logging.info("=" * 60)
            logging.info("ETL Pipeline execution completed successfully!")
            logging.info("=" * 60)

            return {
                "status": "SUCCESS",
                "extracted_records": len(self.extracted_df),
                "transformed_records": len(self.transformed_df),
                "message": "ETL Pipeline completed successfully"
            }

        except Exception as e:
            logging.error("=" * 60)
            logging.error("ETL Pipeline execution failed!")
            logging.error(f"Error: {str(e)}")
            logging.error("=" * 60)
            raise

    def get_extracted_data(self):
        """
        Get the extracted data

        Returns:
            pd.DataFrame: Extracted data
        """
        return self.extracted_df

    def get_transformed_data(self):
        """
        Get the transformed data

        Returns:
            pd.DataFrame: Transformed data
        """
        return self.transformed_df

if __name__ == "__main__":
    # Determine seasons to extract using auto-detection based on current date
    seasons_to_extract = get_seasons(base_year=2023)
    print(f"Seasons to fetch: {seasons_to_extract}")
    
    # Suppress noisy logs from third-party libraries
    import logging as py_logging
    py_logging.getLogger('soccerdata').setLevel(py_logging.WARNING)
    py_logging.getLogger('selenium').setLevel(py_logging.WARNING)
    py_logging.getLogger('urllib3').setLevel(py_logging.WARNING)
    
    try:
        logging.info(f"Starting scheduled ETL run for seasons: {seasons_to_extract}")
        pipeline = ETLPipeline(seasons=seasons_to_extract)
        result = pipeline.run()
        logging.info(f"ETL run completed successfully: {result}")
    except Exception as e:
        logging.error(f"Scheduled ETL run failed: {str(e)}")
        sys.exit(1)