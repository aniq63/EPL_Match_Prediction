import soccerdata as sd
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from src.utils.logger import logging
from src.utils.setting import get_settings


class PremierLeagueStatsAnalyzer:
    """
    Analyzes Premier League player and team statistics using Understat data
    and persists results into Supabase tables.
    """
    
    def __init__(self, season=None):
        try:
            self.season = season if season else self._get_season()
            self.df = self._load_data()
            self.settings = get_settings()
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            raise Exception(f"Initialization failed: {str(e)}")

    # -------------------------
    # Helpers
    # -------------------------
    def _get_season(self):
        now = datetime.utcnow()
        start_year = now.year if now.month >= 8 else now.year - 1
        return f"{start_year}/{start_year + 1}"

    def _load_data(self):
        logging.info(f"Loading Understat data for season {self.season}...")
        understat = sd.Understat(
            leagues="ENG-Premier League",
            seasons=self.season
        )
        return understat.read_player_season_stats().reset_index()

    def _validate_columns(self, columns):
        missing = [col for col in columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def _get_sync_engine(self):
        """
        Create a synchronous SQLAlchemy engine compatible with pandas.to_sql.
        """
        db_url = self.settings.database_url
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        elif db_url.startswith("postgresql+asyncpg://"):
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        
        return create_engine(db_url)

    def _save_to_supabase(self, df, table_name):
        """
        Save a DataFrame to Supabase, replacing the table if it already exists.
        """
        try:
            engine = self._get_sync_engine()
            table_name = table_name.lower()
            logging.info(f"Saving {len(df)} rows to Supabase table: '{table_name}'...")
            
            df.to_sql(
                name=table_name,
                con=engine,
                if_exists='replace',
                index=False
            )
            logging.info(f"Successfully saved to '{table_name}'.")
        except Exception as e:
            logging.error(f"Failed to save to Supabase table '{table_name}': {e}")
            raise

    # -------------------------
    # Analytical Methods
    # -------------------------
    def get_top_players(self, stat, n=5, save=True):
        """
        Generic method for player stats. Saves to 'top_players_{stat}' if save=True.
        """
        try:
            self._validate_columns(['player', 'team', stat])
            result_df = self.df[['player', 'team', stat]] \
                .sort_values(stat, ascending=False).head(n).reset_index(drop=True)
            
            # Add rank column starting from 1
            result_df['rank'] = range(1, len(result_df) + 1)
            
            # Reorder columns to put rank first
            cols = ['rank'] + [c for c in result_df.columns if c != 'rank']
            result_df = result_df[cols]
            
            if save:
                self._save_to_supabase(result_df, f"top_players_{stat}")
                
            return result_df
        except Exception as e:
            logging.error(f"Error in get_top_players({stat}): {e}")
            return pd.DataFrame()

    def get_top_teams(self, stat, n=5, save=True):
        """
        Generic method for team aggregation. Saves to 'top_teams_{stat}' if save=True.
        """
        try:
            self._validate_columns(['team', stat])
            result_df = self.df.groupby('team')[stat] \
                .sum().sort_values(ascending=False).head(n).reset_index()
            
            # Add rank column starting from 1
            result_df['rank'] = range(1, len(result_df) + 1)
            
            # Reorder columns to put rank first
            cols = ['rank'] + [c for c in result_df.columns if c != 'rank']
            result_df = result_df[cols]
            
            if save:
                self._save_to_supabase(result_df, f"top_teams_{stat}")
                
            return result_df
        except Exception as e:
            logging.error(f"Error in get_top_teams({stat}): {e}")
            return pd.DataFrame()

    def get_top_teams_created_chances(self, n=5, save=True):
        """
        Special case: derived metric. Saves to 'top_teams_created_chances' if save=True.
        """
        try:
            self._validate_columns(['team', 'xg_chain', 'xg_buildup'])
            df_copy = self.df.copy()
            # created_chances defined as xg_chain + xg_buildup for this analysis
            df_copy['created_chances'] = df_copy['xg_chain'] + df_copy['xg_buildup']

            result_df = df_copy.groupby('team')['created_chances'] \
                .sum().sort_values(ascending=False).head(n).reset_index()
            
            # Add rank column starting from 1
            result_df['rank'] = range(1, len(result_df) + 1)
            
            # Reorder columns to put rank first
            cols = ['rank'] + [c for c in result_df.columns if c != 'rank']
            result_df = result_df[cols]
            
            if save:
                self._save_to_supabase(result_df, "top_teams_created_chances")
                
            return result_df
        except Exception as e:
            logging.error(f"Error in created chances: {e}")
            return pd.DataFrame()

    def run_all_analyses(self):
        """
        Triggers all analysis methods and refreshes the corresponding Supabase tables.
        """
        logging.info("=" * 60)
        logging.info("STARTING ALL STATS ANALYSES & SUPABASE REFRESH")
        logging.info("=" * 60)
        
        # Player-level stats
        self.get_top_players('goals', save=True)
        self.get_top_players('assists', save=True)
        self.get_top_players('shots', save=True)
        self.get_top_players('key_passes', save=True)
        self.get_top_players('yellow_cards', save=True)
        self.get_top_players('red_cards', save=True)
        
        # Team-level stats
        self.get_top_teams('goals', save=True)
        self.get_top_teams('shots', save=True)
        self.get_top_teams('yellow_cards', save=True)
        self.get_top_teams('red_cards', save=True)
        self.get_top_teams_created_chances(save=True)
        
        logging.info("=" * 60)
        logging.info("ALL STATS ANALYSES COMPLETED SUCCESSFULLY")
        logging.info("=" * 60)


# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    try:
        logging.info("Starting PremierLeagueStatsAnalyzer...")
        analyzer = PremierLeagueStatsAnalyzer()
        analyzer.run_all_analyses()
    except Exception as e:
        logging.error(f"Fatal error in analyzer execution: {e}")