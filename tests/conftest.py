"""Shared pytest fixtures for the EPL Nexus test suite."""

import numpy as np
import pandas as pd
import pytest

from config.constants import ETL_REQUIRED_COLUMNS

# A pool of realistic Premier League team names
TEAMS = [
    "Arsenal", "Chelsea", "Liverpool", "Manchester City",
    "Manchester United", "Tottenham", "Aston Villa", "Everton",
    "Newcastle United", "West Ham", "Brighton", "Fulham",
    "Brentford", "Crystal Palace", "Wolves", "Nottingham Forest",
    "Leicester", "Leeds", "Southampton", "Burnley",
]

BASE_COLS = [
    "date", "home_team", "away_team",
    "home_goals", "away_goals",
    "home_xg", "away_xg",
    "home_ppda", "away_ppda",
    "home_deep_completions", "away_deep_completions",
    "home_points", "away_points",
]


def make_match_matrix(season_start="2024-08-10", rounds=40, rng_seed=7):
    """
    Build a realistic sequential Premier League schedule:
    one match per round per pair, home/away alternating, spanning `rounds` rounds.

    Each round produces len(TEAMS)//2 matches so a team plays once per round.
    The schedule is intentionally deterministic given a seed.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(TEAMS)
    rounds_teams = TEAMS[:]
    start = pd.Timestamp(season_start)

    records = []
    for r in range(rounds):
        if r % 2 == 0:
            rng.shuffle(rounds_teams)
        else:
            rounds_teams = rounds_teams[1:] + rounds_teams[:1]
        for i in range(0, n, 2):
            home, away = rounds_teams[i], rounds_teams[i + 1]
            hg = int(rng.integers(0, 5))
            ag = int(rng.integers(0, 4))
            rec = {
                "date": start + pd.offsets.Day(r * 7),
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
                "home_xg": round(float(rng.uniform(0.4, 2.6)), 3),
                "away_xg": round(float(rng.uniform(0.3, 2.3)), 3),
                "home_ppda": round(float(rng.uniform(5.0, 15.0)), 2),
                "away_ppda": round(float(rng.uniform(5.0, 15.0)), 2),
                "home_deep_completions": int(rng.integers(5, 40)),
                "away_deep_completions": int(rng.integers(5, 40)),
                # 3 = win, 1 = draw, 0 = loss
                "home_points": 3 if hg > ag else (1 if hg == ag else 0),
                "away_points": 3 if ag > hg else (1 if ag == hg else 0),
            }
            records.append(rec)

    df = pd.DataFrame(records)[BASE_COLS]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@pytest.fixture
def raw_matches():
    """A realistic EPL-style match DataFrame (raw, before feature engineering)."""
    return make_match_matrix(rounds=40)


@pytest.fixture
def raw_matches_small():
    """A small raw match DataFrame for quick unit tests (fewer rounds)."""
    return make_match_matrix(season_start="2024-08-10", rounds=14, rng_seed=3)


@pytest.fixture
def feature_columns():
    """The exact 27 engine features the model expects."""
    from config.constants import INPUT_FEATURES

    return list(INPUT_FEATURES)


@pytest.fixture
def required_columns():
    """The columns that ETL data load / transformation guarantees."""
    return list(ETL_REQUIRED_COLUMNS)
