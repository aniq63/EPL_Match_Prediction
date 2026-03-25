# ============================================================
# IMPORTS
# ============================================================

import soccerdata as sd
import pandas as pd
from datetime import datetime
import warnings
from src.utils.logger import logging

warnings.filterwarnings('ignore')


# ============================================================
# DATA EXTRACTION CLASS
# ============================================================

class DataExtraction:
    """
    Extracts team match statistics from Understat for English Premier League seasons.
    """

    def __init__(self, seasons: list[str], league: str = "ENG-Premier League"):
        self.seasons = seasons
        self.league = league

    def extract_pl_data(self) -> pd.DataFrame:
        dfs = []
        failed_seasons = []

        for season in self.seasons:
            try:
                logging.info(f"  Fetching {season}...")
                understat = sd.Understat(leagues=self.league, seasons=season, no_cache = True)
                df_season = understat.read_team_match_stats()

                if df_season.empty:
                    logging.warning(f"No data returned for {season}") 
                    failed_seasons.append(season)
                    continue

                dfs.append(df_season)
                logging.info(f"got {len(df_season)} rows") 

            except Exception as e: 
                logging.error(f"Failed to fetch {season}: {type(e).__name__}: {e}") 
                failed_seasons.append(season)

        if not dfs:
            error_msg = f"No data extracted for any season. Failed seasons: {failed_seasons}"
            logging.error(error_msg)
            raise Exception(error_msg)

        df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Total raw rows: {df.shape[0]} | Columns: {df.shape[1]}") 
        logging.info(f"Columns available: {list(df.columns)}")

        if failed_seasons:
            logging.warning(f"Some seasons failed: {failed_seasons}")

        return df
