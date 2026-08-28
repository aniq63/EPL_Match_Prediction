import os
import sys
from typing import Optional

import mlflow
import mlflow.sklearn
import dagshub
import numpy as np
import pandas as pd
import soccerdata as sd
from datetime import datetime
from dotenv import load_dotenv

from src.utils.setting import get_settings
from src.utils.logger import logging
from src.utils.exception import MyException
from config.constants import MODEL_NAME
from src.components.data_ingestion import DataIngestion
from src.etl.data_transformation import DataTransformation
from sqlalchemy import create_engine

load_dotenv()

MLFLOW_TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

# ── Exact 27 features the model was trained on ────────────────────────────────
FEATURE_COLUMNS = [
    "home_goals_avg_last5",
    "away_goals_avg_last5",
    "home_goals_conceded_avg_last5",
    "away_goals_conceded_avg_last5",
    "home_xg_avg_last5",
    "away_xg_avg_last5",
    "home_ppda_avg_last5",
    "away_ppda_avg_last5",
    "home_deep_completions_avg_last5",
    "away_deep_completions_avg_last5",
    "home_points_last5",
    "away_points_last5",
    "home_team_home_wins_last5",
    "home_team_home_draws_last5",
    "home_team_home_losses_last5",
    "away_team_away_wins_last5",
    "away_team_away_draws_last5",
    "away_team_away_losses_last5",
    "points_diff_last5",
    "goal_diff_avg5",
    "xg_diff_avg5",
    "x_defense_diff",
    "ppda_diff_avg5",
    "deep_comp_diff_avg5",
    "venue_wins_diff",
    "home_venue_advantage",
    "home_advantage",
]

# ── Label Mapping ─────────────────────────────────────────────────────────────
# Based on verified model classes: 0=Draw, 1=Lose, 2=Win
LABEL_MAP = {"0": "Draw", "1": "Lose", "2": "Win"}

# ── ESPN → Understat name map ─────────────────────────────────────────────────
ESPN_TO_UNDERSTAT = {
    "AFC Bournemouth":         "Bournemouth",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Brighton & Hove Albion":  "Brighton",
    "Tottenham Hotspur":       "Tottenham",
    "Nottingham Forest":       "Nottingham Forest",
    "Newcastle United":        "Newcastle United",
    "Manchester United":       "Manchester United",
    "Manchester City":         "Manchester City",
    "West Ham United":         "West Ham",
    "Aston Villa":             "Aston Villa",
    "Crystal Palace":          "Crystal Palace",
    "Leeds United":            "Leeds",
    "Ipswich Town":            "Ipswich",
    "Leicester City":          "Leicester",
    "Sunderland":              "Sunderland",
    "Burnley":                 "Burnley",
    "Brentford":               "Brentford",
    "Fulham":                  "Fulham",
    "Chelsea":                 "Chelsea",
    "Arsenal":                 "Arsenal",
    "Liverpool":               "Liverpool",
    "Everton":                 "Everton",
    "Southampton":             "Southampton",
    "Sheffield United":        "Sheffield United",
    "Luton Town":              "Luton",
    "Nott'm Forest":           "Nottingham Forest",
    "Wolves":                  "Wolverhampton Wanderers",
    "Spurs":                   "Tottenham",
    "Coventry City":           "Coventry City",
    "Hull City":               "Hull City",
    "West Bromwich Albion":    "West Bromwich Albion",
    "Middlesbrough":           "Middlesbrough",
    "Norwich City":            "Norwich",
    "Watford":                 "Watford",
    "Cardiff City":            "Cardiff",
    "Huddersfield Town":       "Huddersfield",
    "Swansea City":            "Swansea",
    "Stoke City":              "Stoke",
    "West Brom":               "West Bromwich Albion",
}


# ── Standalone helpers ────────────────────────────────────────────────────────

def get_current_season() -> str:
    """
    Returns current EPL season string e.g. '2025/2026'.
    EPL seasons start in August:
        Jan–Jul  → previous_year/current_year
        Aug–Dec  → current_year/next_year
    """
    now = datetime.now()
    if now.month >= 8:
        return f"{now.year}/{now.year + 1}"
    return f"{now.year - 1}/{now.year}"


def get_last5_all_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Last 5 matches (home + away combined) for a team.
    Adds unified columns _goals, _goals_con, _xg, _ppda, _deep, _points
    so the same column names work regardless of home/away row.
    """
    home_rows = df[df["home_team"] == team].copy()
    home_rows["_goals"]     = home_rows["home_goals"]
    home_rows["_goals_con"] = home_rows["away_goals"]
    home_rows["_xg"]        = home_rows["home_xg"]
    home_rows["_ppda"]      = home_rows["home_ppda"]
    home_rows["_deep"]      = home_rows["home_deep_completions"]
    home_rows["_points"]    = home_rows["home_points"]

    away_rows = df[df["away_team"] == team].copy()
    away_rows["_goals"]     = away_rows["away_goals"]
    away_rows["_goals_con"] = away_rows["home_goals"]
    away_rows["_xg"]        = away_rows["away_xg"]
    away_rows["_ppda"]      = away_rows["away_ppda"]
    away_rows["_deep"]      = away_rows["away_deep_completions"]
    away_rows["_points"]    = away_rows["away_points"]

    combined = pd.concat([home_rows, away_rows]).sort_values("date").tail(5)

    if len(combined) < 1:
        raise ValueError(f"No match history found for '{team}'.")
    return combined


def get_last5_home_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Last 5 HOME-only matches for a team (for venue form)."""
    rows = df[df["home_team"] == team].sort_values("date").tail(5)
    if len(rows) < 1:
        raise ValueError(f"No home match history found for '{team}'.")
    return rows


def get_last5_away_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Last 5 AWAY-only matches for a team (for venue form)."""
    rows = df[df["away_team"] == team].sort_values("date").tail(5)
    if len(rows) < 1:
        raise ValueError(f"No away match history found for '{team}'.")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
class PredictionPipeline:
    """
    End-to-end EPL gameweek prediction pipeline.
    """

    # ── Class-level caches — survive across calls in the same process ──────────
    _cached_model    = None   # model object loaded from S3
    _cached_clean_df = None   # transformed DataFrame for current season

    def __init__(self, stage: str = "Production"):
        self.stage      = stage
        self.model_name = MODEL_NAME
        self.model_uri  = f"models:/{self.model_name}/{self.stage}"

        try:
            self._connect_to_mlflow()
            logging.info(
                f"PredictionPipeline initialised — will load '{self.model_uri}' "
                f"from the MLflow registry on DagsHub."
            )
        except Exception as e:
            logging.error(f"MLflow connection init failed: {e}")
            raise MyException(e, sys)

    # ══════════════════════════════════════════════════════════════════════════
    # MLflow / DagsHub connection
    # ══════════════════════════════════════════════════════════════════════════
    def _connect_to_mlflow(self):
        """Configure MLflow to talk to the DagsHub-hosted tracking/registry server."""
        try:
            uri = MLFLOW_TRACKING_URI or \
                  "https://dagshub.com/aniqramzan5758/EPL_Match_Prediction.mlflow"

            username = MLFLOW_TRACKING_USERNAME
            password = MLFLOW_TRACKING_PASSWORD or os.getenv("DAGSHUB_TOKEN")

            if username and password:
                os.environ["MLFLOW_TRACKING_USERNAME"] = username
                os.environ["MLFLOW_TRACKING_PASSWORD"] = password

                if "dagshub.com" in uri.lower():
                    try:
                        parts = uri.split("/")
                        if len(parts) >= 5:
                            repo_owner = parts[3]
                            repo_name  = parts[4].replace(".mlflow", "")

                            logging.info(f"Initializing DagsHub for {repo_owner}/{repo_name}")
                            os.environ["DAGSHUB_TOKEN"]      = password
                            os.environ["DAGSHUB_USER_TOKEN"] = password

                            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
                    except Exception as e:
                        logging.warning(f"DagsHub init failed, falling back to standard MLflow: {e}")

            mlflow.set_tracking_uri(uri)
            logging.info(f"Connected to MLflow at: {uri}")

        except Exception as e:
            logging.error(f"Error configuring MLflow: {e}")
            raise MyException(e, sys)

    # ══════════════════════════════════════════════════════════════════════════
    # FIXTURE SOURCE — fetch upcoming EPL fixtures from ESPN
    # ══════════════════════════════════════════════════════════════════════════
    def _fetch_espn_fixtures(self, league: str = "ENG-Premier League") -> pd.DataFrame:
        """
        Fetch the next full Premier League gameweek from ESPN.

        A standard EPL gameweek = 10 matches, with each of the 20 clubs
        playing exactly once. Starting from the earliest upcoming fixture,
        this walks forward through the sorted fixture list and keeps adding
        matches until either:
          - 10 matches have been collected (a complete gameweek), or
          - a club that has already appeared would play a second time
            (a sign the *next* gameweek's fixtures have started, e.g. because
            of midweek re-arranged games) — in which case we stop early and
            return only the matches collected so far.

        This is safer than a fixed date window because gameweeks aren't
        always confined to a neat few-day span (rearranged/rescheduled
        fixtures can spill outside a fixed window).

        Returns
        -------
        pd.DataFrame  columns: home_team, away_team, date
                      Team names are ESPN format — not yet mapped to Understat.
        """
        try:
            now    = datetime.utcnow()
            season = now.year + 1 if now.month >= 8 else now.year

            logging.info(f"Fetching fixtures from ESPN (season={season})...")
            espn = sd.ESPN(leagues=league, seasons=season)

            df = (
                espn.read_schedule()
                .assign(date=lambda df: pd.to_datetime(df["date"]))
                .sort_values("date")
            )

            # Filter for future matches
            upcoming = df[df["date"] > pd.Timestamp.utcnow()].reset_index(drop=True)

            if upcoming.empty:
                logging.warning("No future fixtures found in ESPN data.")
                return pd.DataFrame(columns=["home_team", "away_team", "date"])

            # 🧠 Walk forward and build one complete gameweek: 10 matches,
            # each of the 20 clubs appearing exactly once.
            MAX_GAMEWEEK_MATCHES = 10
            seen_teams   = set()
            selected     = []

            for _, row in upcoming.iterrows():
                home_team = row["home_team"]
                away_team = row["away_team"]

                if home_team in seen_teams or away_team in seen_teams:
                    # This match belongs to a later gameweek — stop here.
                    break

                selected.append(row)
                seen_teams.add(home_team)
                seen_teams.add(away_team)

                if len(selected) == MAX_GAMEWEEK_MATCHES:
                    break

            fixtures = (
                pd.DataFrame(selected)[["home_team", "away_team", "date"]]
                .reset_index(drop=True)
            )

            logging.info(
                f"Selected gameweek: {len(fixtures)} match(es) covering "
                f"{len(seen_teams)} club(s)."
            )
            if len(fixtures) < MAX_GAMEWEEK_MATCHES:
                logging.warning(
                    f"Gameweek has only {len(fixtures)}/{MAX_GAMEWEEK_MATCHES} matches — "
                    f"this may be a partial gameweek (postponements, midweek fixtures "
                    f"spilling into the next round, or season start/end)."
                )

            return fixtures

        except Exception as e:
            logging.error(f"ESPN fetch failed: {e}")
            raise MyException(e, sys)

    # ══════════════════════════════════════════════════════════════════════════
    # NAME MAPPING — ESPN name → Understat name
    # ══════════════════════════════════════════════════════════════════════════
    def map_espn_name(self, espn_name: str) -> Optional[str]:
        """
        Convert an ESPN team name to its Understat equivalent.
        Returns None if not in the map — caller skips the fixture.
        Add missing entries to ESPN_TO_UNDERSTAT at the top of this file.
        """
        if espn_name in ESPN_TO_UNDERSTAT:
            return ESPN_TO_UNDERSTAT[espn_name]
        logging.warning(f"UNMAPPED team: '{espn_name}' — add to ESPN_TO_UNDERSTAT.")
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Fetch + clean current season data  (cached after first call)
    # ══════════════════════════════════════════════════════════════════════════
    def fetch_and_clean_data(self) -> pd.DataFrame:
        """
        Fetch ALL historical data from the warehouse (not just the current
        season) and run DataTransformation. Result is cached so repeated
        calls are free.

        Why all history and not just the current season: rolling "last 5"
        features need 5 prior matches per team. Early in a new season
        (gameweeks 1-4), a team simply hasn't played 5 home/away matches
        yet *this season* — but its form absolutely still exists from the
        end of last season. Pulling the full history lets the last-5 slice
        naturally span the season boundary instead of failing with
        "No match history found" for the first few gameweeks of every season.

        Returns
        -------
        pd.DataFrame — cleaned data, date column as datetime, all seasons,
                       sorted chronologically, restricted to matches that
                       have already been played (date <= now).
        """
        # Return cache if available
        if PredictionPipeline._cached_clean_df is not None:
            logging.info("Using cached clean DataFrame.")
            return PredictionPipeline._cached_clean_df

        try:
            logging.info("Fetching full match history from the warehouse...")

            ingestion = DataIngestion()
            raw_df    = ingestion.fetch_all_data()
            logging.info(f"Fetched {len(raw_df)} rows across all seasons.")

            if raw_df.empty:
                raise ValueError("Warehouse returned no data — cannot build features.")

            # Keep only matches that have actually been played, sorted chronologically.
            raw_df["date"] = pd.to_datetime(raw_df["date"])
            raw_df = raw_df[raw_df["date"] <= pd.Timestamp.now()].sort_values("date")

            transformer = DataTransformation(raw_df)
            clean_df    = transformer.transform_pl_data()
            clean_df["date"] = pd.to_datetime(clean_df["date"])

            # Cache for subsequent fixtures in the same run
            PredictionPipeline._cached_clean_df = clean_df

            logging.info(f"Data cleaned and cached. Shape: {clean_df.shape}")
            return clean_df

        except Exception as e:
            logging.error(f"fetch_and_clean_data failed: {e}")
            raise MyException(e, sys)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Get last-5 slices for both teams
    # ══════════════════════════════════════════════════════════════════════════
    def get_last5_for_teams(
        self,
        clean_df: pd.DataFrame,
        home_team: str,
        away_team: str,
    ) -> dict:
        """
        Slice the cleaned DataFrame into 4 views needed for features:
            home_all  — last 5 all matches for home team  → rolling averages
            away_all  — last 5 all matches for away team  → rolling averages
            home_h5   — last 5 HOME-only for home team    → venue form
            away_a5   — last 5 AWAY-only for away team    → venue form

        Raises ValueError if a team doesn't have enough history.
        Returns dict { "home_all", "away_all", "home_h5", "away_a5" }
        """
        try:
            logging.info(f"Slicing last-5: {home_team} (home) | {away_team} (away)")

            home_all = get_last5_all_matches(clean_df, home_team)
            away_all = get_last5_all_matches(clean_df, away_team)
            home_h5  = get_last5_home_matches(clean_df, home_team)
            away_a5  = get_last5_away_matches(clean_df, away_team)

            return {
                "home_all": home_all,
                "away_all": away_all,
                "home_h5":  home_h5,
                "away_a5":  away_a5,
            }

        except ValueError:
            raise   # bubble up "not enough history" to the fixture loop
        except Exception as e:
            logging.error(f"get_last5_for_teams failed: {e}")
            raise MyException(e, sys)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3a — Home team rolling averages
    # ══════════════════════════════════════════════════════════════════════════
    def build_home_features(self, home_all: pd.DataFrame) -> dict:
        """Rolling averages + points sum for the home team (last 5 matches)."""
        try:
            return {
                "goals_avg":     round(float(home_all["_goals"].mean()), 4),
                "goals_con_avg": round(float(home_all["_goals_con"].mean()), 4),
                "xg_avg":        round(float(home_all["_xg"].mean()), 4),
                "ppda_avg":      round(float(home_all["_ppda"].mean()), 4),
                "deep_avg":      round(float(home_all["_deep"].mean()), 4),
                "points_sum":    int(home_all["_points"].sum()),
            }
        except Exception as e:
            logging.error(f"build_home_features failed: {e}")
            raise MyException(e, sys)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3b — Away team rolling averages
    # ══════════════════════════════════════════════════════════════════════════
    def build_away_features(self, away_all: pd.DataFrame) -> dict:
        """Rolling averages + points sum for the away team (last 5 matches)."""
        try:
            return {
                "goals_avg":     round(float(away_all["_goals"].mean()), 4),
                "goals_con_avg": round(float(away_all["_goals_con"].mean()), 4),
                "xg_avg":        round(float(away_all["_xg"].mean()), 4),
                "ppda_avg":      round(float(away_all["_ppda"].mean()), 4),
                "deep_avg":      round(float(away_all["_deep"].mean()), 4),
                "points_sum":    int(away_all["_points"].sum()),
            }
        except Exception as e:
            logging.error(f"build_away_features failed: {e}")
            raise MyException(e, sys)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3c — Venue-specific form features
    # ══════════════════════════════════════════════════════════════════════════
    def build_venue_features(
        self,
        home_h5: pd.DataFrame,
        away_a5: pd.DataFrame,
    ) -> dict:
        """
        Venue-specific win/draw/loss counts.
            home_h5 → home team's last 5 HOME matches  (check home_points col)
            away_a5 → away team's last 5 AWAY matches  (check away_points col)
        Points encoding: 3 = win, 1 = draw, 0 = loss
        """
        try:
            home_wins   = int((home_h5["home_points"] == 3).sum())
            home_draws  = int((home_h5["home_points"] == 1).sum())
            home_losses = int((home_h5["home_points"] == 0).sum())

            away_wins   = int((away_a5["away_points"] == 3).sum())
            away_draws  = int((away_a5["away_points"] == 1).sum())
            away_losses = int((away_a5["away_points"] == 0).sum())

            return {
                "home_venue_wins":      home_wins,
                "home_venue_draws":     home_draws,
                "home_venue_losses":    home_losses,
                "away_venue_wins":      away_wins,
                "away_venue_draws":     away_draws,
                "away_venue_losses":    away_losses,
                "home_venue_advantage": round(home_wins / max(len(home_h5), 1), 4),
                "venue_wins_diff":      home_wins - away_wins,
            }
        except Exception as e:
            logging.error(f"build_venue_features failed: {e}")
            raise MyException(e, sys)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — Assemble the 27-feature prediction row
    # ══════════════════════════════════════════════════════════════════════════
    def build_prediction_row(
        self,
        home_feats: dict,
        away_feats: dict,
        venue_feats: dict,
    ) -> pd.DataFrame:
        """
        Combine all features into one 27-column DataFrame row in the exact
        column order the trained model expects.

        Returns pd.DataFrame  shape (1, 27)
        """
        try:
            hf = home_feats
            af = away_feats
            vf = venue_feats

            row = {
                # Rolling averages
                "home_goals_avg_last5":            hf["goals_avg"],
                "away_goals_avg_last5":            af["goals_avg"],
                "home_goals_conceded_avg_last5":   hf["goals_con_avg"],
                "away_goals_conceded_avg_last5":   af["goals_con_avg"],
                "home_xg_avg_last5":               hf["xg_avg"],
                "away_xg_avg_last5":               af["xg_avg"],
                "home_ppda_avg_last5":             hf["ppda_avg"],
                "away_ppda_avg_last5":             af["ppda_avg"],
                "home_deep_completions_avg_last5": hf["deep_avg"],
                "away_deep_completions_avg_last5": af["deep_avg"],
                # Points totals (last 5 matches)
                "home_points_last5":               hf["points_sum"],
                "away_points_last5":               af["points_sum"],
                # Venue form counts
                "home_team_home_wins_last5":        vf["home_venue_wins"],
                "home_team_home_draws_last5":       vf["home_venue_draws"],
                "home_team_home_losses_last5":      vf["home_venue_losses"],
                "away_team_away_wins_last5":        vf["away_venue_wins"],
                "away_team_away_draws_last5":       vf["away_venue_draws"],
                "away_team_away_losses_last5":      vf["away_venue_losses"],
                # Diff features (home − away)
                "points_diff_last5":    hf["points_sum"]    - af["points_sum"],
                "goal_diff_avg5":       hf["goals_avg"]     - af["goals_avg"],
                "xg_diff_avg5":         hf["xg_avg"]        - af["xg_avg"],
                "x_defense_diff":       af["goals_con_avg"] - hf["goals_con_avg"],
                "ppda_diff_avg5":       hf["ppda_avg"]      - af["ppda_avg"],
                "deep_comp_diff_avg5":  hf["deep_avg"]      - af["deep_avg"],
                # Venue level
                "venue_wins_diff":       vf["venue_wins_diff"],
                "home_venue_advantage":  vf["home_venue_advantage"],
                # Always 1 for a future home match
                "home_advantage": 1,
            }

            # Enforce exact column order the model was trained with
            return pd.DataFrame([row])[FEATURE_COLUMNS]

        except Exception as e:
            logging.error(f"build_prediction_row failed: {e}")
            raise MyException(e, sys)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Load model from MLflow Model Registry (DagsHub) — cached after first call
    # ══════════════════════════════════════════════════════════════════════════
    def load_model(self):
        """
        Load the model straight from the MLflow Model Registry (hosted on
        DagsHub) at `models:/{model_name}/{stage}`, then serve from an
        in-memory cache for subsequent calls in the same process.
        """
        if PredictionPipeline._cached_model is not None:
            logging.info("Using cached model.")
            return PredictionPipeline._cached_model

        try:
            logging.info(f"Loading model from MLflow registry: {self.model_uri}")
            model = mlflow.sklearn.load_model(self.model_uri)

            PredictionPipeline._cached_model = model
            logging.info("Model loaded from MLflow and cached.")
            return model

        except Exception as e:
            logging.error(f"load_model failed: {e}")
            raise MyException(e, sys)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — Run inference
    # ══════════════════════════════════════════════════════════════════════════
    def predict(
        self,
        prediction_row: pd.DataFrame,
        home_team: str,
        away_team: str,
    ) -> dict:
        """
        Run model.predict() + predict_proba() and return a clean result dict.

        Label handling
        --------------
        AdaBoostClassifier with LabelEncoder stores string labels directly in
        model.classes_ — sorted alphabetically by sklearn:
            model.classes_ = ["Draw", "Lose", "Win"]   (indices 0, 1, 2)

        model.predict() already returns the decoded string label ("Draw" / "Lose" / "Win")
        because your ModelTrainer uses LabelEncoder and the pipeline stores the
        inverse-transformed classes. So raw_pred is ALREADY a string — no
        manual index-to-label mapping is needed.

        model.predict_proba() returns probabilities in the SAME order as
        model.classes_, so zipping them together gives the correct pairing.

        Returns
        -------
        {
            "home_team":        "Arsenal",
            "away_team":        "Chelsea",
            "predicted_result": "Win",        ← home team's perspective
                                                Win  = home team wins
                                                Draw = draw
                                                Lose = home team loses
            "probabilities": {
                "Draw": 27.1,   ← index 0 in model.classes_
                "Lose": 18.7,   ← index 1 in model.classes_
                "Win":  54.2,   ← index 2 in model.classes_
            },
            "confidence": 54.2,   ← max probability
            "stage":      "Production",
        }
        """
        try:
            model = self.load_model()

            # ── Probabilities ─────────────────────────────────────────────────
            # model.predict_proba() → array shape (1, 3)
            # model.classes_        → [0, 1, 2]
            proba       = model.predict_proba(prediction_row)[0]   # shape (3,)
            class_names = [str(c) for c in model.classes_]         # ["0", "1", "2"]

            # Map numeric class names to human-readable labels
            prob_dict = {
                LABEL_MAP.get(cls, cls): round(float(p) * 100, 1)
                for cls, p in zip(class_names, proba)
            }

            # ── Predict ───────────────────────────────────────────────────────
            # model.predict() returns numeric index
            raw_pred        = str(model.predict(prediction_row)[0])
            predicted_label = LABEL_MAP.get(raw_pred, raw_pred)

            # Sanity check — predicted label must be one of the known classes
            if predicted_label not in prob_dict:
                raise ValueError(
                    f"predicted_label '{predicted_label}' not in mapped classes: "
                    f"{list(prob_dict.keys())}. Check LabelMapping consistency."
                )

            result = {
                "home_team":        home_team,
                "away_team":        away_team,
                "predicted_result": predicted_label,
                "probabilities":    prob_dict,
                "confidence":       round(prob_dict[predicted_label], 1),
                "stage":            self.stage,
            }

            logging.info(
                f"  → {home_team} vs {away_team}: {predicted_label} "
                f"(Win={prob_dict.get('Win', 0.0)}% | "
                f"Draw={prob_dict.get('Draw', 0.0)}% | "
                f"Lose={prob_dict.get('Lose', 0.0)}%)"
            )
            return result

        except Exception as e:
            logging.error(f"predict failed: {e}")
            raise MyException(e, sys)

    # ══════════════════════════════════════════════════════════════════════════
    # Feature Importance
    # ══════════════════════════════════════════════════════════════════════════
    def get_feature_importance(self) -> list:
        """
        Extract feature importance from the loaded model.
        Returns the top 8 most important features paired with their scores.
        """
        try:
            model = self.load_model()
            
            # AdaBoost models typically have feature_importances_
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                
                # Pair with FEATURE_COLUMNS
                feat_imp = [
                    {"feature": feat, "importance": round(float(imp), 4)}
                    for feat, imp in zip(FEATURE_COLUMNS, importances)
                ]
                
                # Sort by importance descending
                feat_imp.sort(key=lambda x: x["importance"], reverse=True)
                
                # Take top 8
                return feat_imp[:8]
            else:
                logging.warning("Model does not have feature_importances_ attribute.")
                return []
        except Exception as e:
            logging.error(f"get_feature_importance failed: {e}")
            return []

    # ══════════════════════════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════════════════════════
    @classmethod
    def clear_cache(cls):
        """Force fresh model + data download on next call."""
        cls._cached_model    = None
        cls._cached_clean_df = None
        logging.info("All caches cleared.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main() -> dict:
    """
    Predict the full upcoming EPL gameweek.

    Returns
    -------
    dict keyed by "HomeTeam vs AwayTeam" (ESPN display names):
    {
        "Arsenal vs Chelsea": {
            "home_team":        "Arsenal",       ← Understat name used for features
            "away_team":        "Chelsea",
            "match_date":       "2026-04-05",
            "predicted_result": "Win",
            "probabilities":    {"Draw": 27.1, "Lose": 18.7, "Win": 54.2},
            "confidence":       54.2,
            "stage":            "Production",
            "home_team_espn":   "Arsenal",       ← original ESPN name
            "away_team_espn":   "Chelsea",
        },
        ...
        "Nott'm Forest vs Wolves": {
            "predicted_result": "SKIPPED",
            "error": "Team name not in ESPN_TO_UNDERSTAT map.",
        },
    }
    """
    logging.info("=" * 60)
    logging.info("GAMEWEEK PREDICTION PIPELINE START")
    logging.info("=" * 60)

    pipeline    = PredictionPipeline(stage="Production")
    all_results = {}

    # ── Step A: fetch upcoming fixtures from ESPN ──────────────────────────────
    fixtures = pipeline._fetch_espn_fixtures()

    if fixtures.empty:
        logging.warning("No upcoming fixtures found. Exiting.")
        return all_results

    logging.info(f"Found {len(fixtures)} upcoming fixture(s):")
    for _, f in fixtures.iterrows():
        logging.info(f"  {f['home_team']} vs {f['away_team']}  [{f['date'].date()}]")

    # ── Step B: fetch + clean season data ONCE (shared by all fixtures) ────────
    clean_df = pipeline.fetch_and_clean_data()

    # ── Step C: loop through each fixture and predict ──────────────────────────
    for _, fixture in fixtures.iterrows():

        espn_home  = fixture["home_team"]
        espn_away  = fixture["away_team"]
        match_date = str(fixture["date"].date())
        match_key  = f"{espn_home} vs {espn_away}"

        logging.info("-" * 50)
        logging.info(f"Processing: {match_key}  [{match_date}]")

        # Map ESPN names → Understat names
        home_us = pipeline.map_espn_name(espn_home)
        away_us = pipeline.map_espn_name(espn_away)

        if home_us is None or away_us is None:
            logging.warning(f"SKIPPED: {match_key} — unmapped name.")
            all_results[match_key] = {
                "home_team":        espn_home,
                "away_team":        espn_away,
                "match_date":       match_date,
                "predicted_result": "SKIPPED",
                "error":            "Team name not in ESPN_TO_UNDERSTAT map.",
            }
            continue

        try:
            # Step 2: last-5 slices
            slices = pipeline.get_last5_for_teams(clean_df, home_us, away_us)

            # Step 3a: home rolling averages
            home_feats = pipeline.build_home_features(slices["home_all"])

            # Step 3b: away rolling averages
            away_feats = pipeline.build_away_features(slices["away_all"])

            # Step 3c: venue form
            venue_feats = pipeline.build_venue_features(slices["home_h5"], slices["away_a5"])

            # Step 4: 27-feature row
            prediction_row = pipeline.build_prediction_row(home_feats, away_feats, venue_feats)

            # Step 5 + 6: load model → predict
            result = pipeline.predict(prediction_row, home_us, away_us)

            # Attach display metadata
            result["match_date"]     = match_date
            result["home_team_espn"] = espn_home
            result["away_team_espn"] = espn_away

            all_results[match_key] = result

        except ValueError as ve:
            # Not enough history for this team — skip, don't crash the whole loop
            logging.warning(f"  Could not predict {match_key}: {ve}")
            all_results[match_key] = {
                "home_team":        espn_home,
                "away_team":        espn_away,
                "match_date":       match_date,
                "predicted_result": "ERROR",
                "error":            str(ve),
            }

        except MyException as me:
            logging.error(f"  Pipeline error for {match_key}: {me}")
            all_results[match_key] = {
                "home_team":        espn_home,
                "away_team":        espn_away,
                "match_date":       match_date,
                "predicted_result": "ERROR",
                "error":            str(me),
            }

    # ── Summary log ───────────────────────────────────────────────────────────
    valid     = [r for r in all_results.values() if r["predicted_result"] not in ("ERROR", "SKIPPED")]
    home_wins = sum(1 for r in valid if r["predicted_result"] == "Win")
    draws     = sum(1 for r in valid if r["predicted_result"] == "Draw")
    away_wins = sum(1 for r in valid if r["predicted_result"] == "Lose")

    logging.info("=" * 60)
    logging.info("GAMEWEEK PREDICTION RESULTS")
    logging.info("=" * 60)
    logging.info(f"  {'Match':<40}  {'Date':<12}  {'Result':>6}  {'Conf%':>6}")
    logging.info(f"  {'-' * 64}")

    for match_key, res in all_results.items():
        predicted = res.get("predicted_result", "?")
        conf      = res.get("confidence")
        label     = {"Win": "HOME", "Draw": "DRAW", "Lose": "AWAY"}.get(predicted, predicted)
        conf_str  = f"{conf:.1f}" if conf is not None else "  —"
        logging.info(f"  {match_key:<40}  {res.get('match_date',''):<12}  {label:>6}  {conf_str:>6}")

    logging.info("-" * 64)
    logging.info(f"  Predicted {len(valid)}/{len(all_results)} | Home={home_wins} Draw={draws} Away={away_wins}")
    logging.info("=" * 60)

    # ── Step D: Save results to Supabase ──────────────────────────────────────
    if valid:
        try:
            logging.info("Converting predictions to DataFrame for Supabase...")
            # Prepare rows, flattening the 'probabilities' dict
            rows = []
            for res in valid:
                flat_row = {
                    "home_team":        res["home_team_espn"],
                    "away_team":        res["away_team_espn"],
                    "match_date":       res["match_date"],
                    "predicted_result": res["predicted_result"],
                    "confidence":       res["confidence"],
                }
                # Add probabilities as individual columns
                for outcome, prob in res["probabilities"].items():
                    flat_row[f"prob_{outcome.lower()}"] = prob
                
                rows.append(flat_row)
            
            payload_df = pd.DataFrame(rows)
            
            # Sync DB connection for pandas to_sql
            settings = get_settings()
            db_url   = settings.database_url
            if db_url.startswith("postgresql+asyncpg://"):
                db_url = db_url.replace("postgresql+asyncpg://", "postgresql://", 1)
            elif db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            
            sync_engine = create_engine(db_url)
            
            logging.info(f"Saving {len(payload_df)} predictions to Supabase table 'gameweek_predictions'...")
            payload_df.to_sql(
                name="gameweek_predictions",
                con=sync_engine,
                if_exists="replace",
                index=False
            )
            logging.info("Supabase upload successful (table replaced).")

            # ── Step E: Save feature importance to Supabase ───────────────────────────
            try:
                logging.info("Extracting top 8 important features...")
                top_features = pipeline.get_feature_importance()
                
                if top_features:
                    feat_df = pd.DataFrame(top_features)
                    feat_df["created_at"] = datetime.now()
                    
                    logging.info(f"Saving top 8 features to Supabase table 'feature_importance'...")
                    feat_df.to_sql(
                        name="feature_importance",
                        con=sync_engine,
                        if_exists="replace",
                        index=False
                    )
                    logging.info("Feature importance upload successful (table replaced).")
                else:
                    logging.warning("No feature importance found to save.")
            except Exception as feat_err:
                logging.error(f"Failed to save feature importance to Supabase: {feat_err}")

        except Exception as db_err:
            logging.error(f"Failed to save predictions to Supabase: {db_err}")

    logging.info("GAMEWEEK PREDICTION PIPELINE COMPLETE")

    return all_results


if __name__ == "__main__":
    results = main()
    print(results)