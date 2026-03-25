"""
Configuration and Constants for EPL Match Prediction Project

This file centralizes all constant values and configuration settings
used throughout the project to ensure consistency and ease of updates.
"""

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# Table name for storing Premier League match data
TABLE_NAME = "epl_matches"

# ============================================================================
# ETL PIPELINE CONFIGURATION
# ============================================================================

# Required columns for transformed data before loading to database
ETL_REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
    "home_np_xg",
    "away_np_xg",
    "home_ppda",
    "away_ppda",
    "home_deep_completions",
    "away_deep_completions",
    "home_points",
    "away_points",
]

# Columns to drop during data transformation
ETL_COLUMNS_TO_DROP = [
    "league_id",
    "season_id",
    "game_id",
    "home_team_id",
    "away_team_id",
    "home_team_code",
    "away_team_code",
    "home_expected_points",
    "away_expected_points",
    "home_np_xg_difference",
    "away_np_xg_difference",
]

# Base season for continuous data extraction
ETL_BASE_SEASON = 2023

# ============================================================================
# DATA SOURCE CONFIGURATION
# ============================================================================

# Football data source (using soccerdata library)
DATA_SOURCE_LEAGUE = "ENG-Premier League"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Maximum log file size in bytes (10 MB)
MAX_LOG_SIZE = 10 * 1024 * 1024

# Number of backup log files to keep
LOG_BACKUP_COUNT = 5

# Log file name
LOG_FILE_NAME = "app.log"
