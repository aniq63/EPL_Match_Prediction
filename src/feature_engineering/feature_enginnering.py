import pandas as pd
from src.utils.logger import logging

class RowTracker:
    def __init__(self):
        self.log = []
        self._current = None

    def before(self, step_name: str, df: pd.DataFrame):
        self._current = {"step": step_name, "before": len(df)}

    def after(self, df: pd.DataFrame, note: str = ""):
        assert self._current is not None, "Call .before() first"
        entry = {
            **self._current,
            "after": len(df),
            "dropped": self._current["before"] - len(df),
            "note": note,
        }
        self.log.append(entry)

        dropped = entry["dropped"]
        flag = " ROWS DROPPED" if dropped > 0 else ""

        logging.info(f"[{entry['step']}] {entry['before']} -> {entry['after']} "
                     f"(dropped {dropped}){flag}")

        self._current = None

    def print_audit(self):
        logging.info("\n" + "=" * 60)
        logging.info("PIPELINE ROW AUDIT")
        logging.info("=" * 60)
        for e in self.log:
            logging.info(f"{e['step']}: {e['before']} -> {e['after']} "
                         f"(lost {e['dropped']})")
            if e["note"]:
                logging.info(f"  |_ {e['note']}")
        logging.info("=" * 60)


# ============================================================
# Feature Engineering Class
# ============================================================

class FeatureEngineering:

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        self.df = df.copy()
        self.tracker = RowTracker()

        logging.info(f"[INIT] Data shape: {self.df.shape}")

    # ============================================================
    # STEP 1 — Basic Features
    # ============================================================
    def basic_features(self):
        try:
            self.tracker.before("feature_engineering", self.df)

            df = self.df
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)

            df["result"] = df.apply(
                lambda r: "Win" if r["home_goals"] > r["away_goals"]
                else "Draw" if r["home_goals"] == r["away_goals"]
                else "Lose",
                axis=1
            )

            df["home_goals_conceded"] = df["away_goals"]
            df["away_goals_conceded"] = df["home_goals"]

            logging.info("Result distribution:")
            logging.info(df["result"].value_counts())

            self.df = df
            self.tracker.after(self.df, "no rows should be dropped here")
            return self

        except Exception as e:
            raise RuntimeError(f"Basic feature step failed: {e}")

    # ============================================================
    # STEP 2 — Rolling Averages (last 5, unified team timeline)
    # ============================================================
    def rolling_features(self):
        try:
            self.tracker.before("rolling_averages", self.df)

            df = self.df

            STAT_COLS = [
                "goals", "goals_conceded",
                "xg", "ppda", "deep_completions",
            ]

            # ---- Build unified team timeline ----
            home = df[[
                "date", "home_team",
                "home_goals", "home_goals_conceded",
                "home_xg", "home_ppda", "home_deep_completions",
            ]].rename(columns={
                "home_team": "team",
                "home_goals": "goals",
                "home_goals_conceded": "goals_conceded",
                "home_xg": "xg",
                "home_ppda": "ppda",
                "home_deep_completions": "deep_completions",
            })
            home["venue"] = "home"

            away = df[[
                "date", "away_team",
                "away_goals", "away_goals_conceded",
                "away_xg", "away_ppda", "away_deep_completions",
            ]].rename(columns={
                "away_team": "team",
                "away_goals": "goals",
                "away_goals_conceded": "goals_conceded",
                "away_xg": "xg",
                "away_ppda": "ppda",
                "away_deep_completions": "deep_completions",
            })
            away["venue"] = "away"

            tl = pd.concat([home, away], ignore_index=True)
            tl.sort_values(["team", "date"], inplace=True)

            # ---- Rolling averages ----
            grp = tl.groupby("team")
            for col in STAT_COLS:
                tl[f"{col}_avg_last5"] = (
                    grp[col]
                    .rolling(5, closed="left")
                    .mean()
                    .reset_index(0, drop=True)
                )

            avg_cols = [f"{c}_avg_last5" for c in STAT_COLS]
            before = len(tl)
            tl.dropna(subset=avg_cols, inplace=True)
            logging.info(f"Timeline: {before} -> {len(tl)} (rolling drop)")

            # ---- Merge back: LEFT join then explicit NaN drop ----
            keep = ["date", "team"] + avg_cols

            home_avgs = tl[tl["venue"] == "home"][keep].rename(
                columns={"team": "home_team",
                         **{c: f"home_{c}" for c in avg_cols}}
            )
            away_avgs = tl[tl["venue"] == "away"][keep].rename(
                columns={"team": "away_team",
                         **{c: f"away_{c}" for c in avg_cols}}
            )

            n0 = len(df)
            df = df.merge(home_avgs, on=["date", "home_team"], how="left")
            df = df.merge(away_avgs, on=["date", "away_team"], how="left")
            assert len(df) == n0, f"Rolling merge changed row count {n0} -> {len(df)}"

            drop_cols = ([f"home_{c}" for c in avg_cols] +
                         [f"away_{c}" for c in avg_cols])
            before = len(df)
            df.dropna(subset=drop_cols, inplace=True)
            logging.info(f"Matches dropped (rolling window): {before - len(df)}")

            self.df = df
            self.tracker.after(self.df, "dropped early matches (<5 games)")
            return self

        except Exception as e:
            raise RuntimeError(f"Rolling feature step failed: {e}")

    # ============================================================
    # STEP 3 — Points last 5
    # ============================================================
    def points_last5(self):
        try:
            self.tracker.before("points_last5", self.df)

            df = self.df

            home = df[["date", "home_team", "home_points"]].rename(
                columns={"home_team": "team", "home_points": "points"}
            )
            away = df[["date", "away_team", "away_points"]].rename(
                columns={"away_team": "team", "away_points": "points"}
            )

            tm = pd.concat([home, away], ignore_index=True)
            tm.sort_values(["team", "date"], inplace=True)

            tm["pts_last5"] = (
                tm.groupby("team")["points"]
                .rolling(5, closed="left")
                .sum()
                .reset_index(0, drop=True)
            )

            tm.dropna(subset=["pts_last5"], inplace=True)
            pts_cols = ["date", "team", "pts_last5"]
            n0 = len(df)

            df = df.merge(
                tm[pts_cols], left_on=["date", "home_team"],
                right_on=["date", "team"], how="left"
            ).rename(columns={"pts_last5": "home_points_last5"}).drop(columns="team")

            assert len(df) == n0, \
                f"Home points merge changed row count {n0} -> {len(df)}. Duplicate keys?"

            df = df.merge(
                tm[pts_cols], left_on=["date", "away_team"],
                right_on=["date", "team"], how="left"
            ).rename(columns={"pts_last5": "away_points_last5"}).drop(columns="team")

            assert len(df) == n0, \
                f"Away points merge changed row count {n0} -> {len(df)}. Duplicate keys?"

            before = len(df)
            df.dropna(subset=["home_points_last5", "away_points_last5"], inplace=True)
            logging.info(f"Points drop: {before - len(df)}")

            self.df = df
            self.tracker.after(self.df)
            return self

        except Exception as e:
            raise RuntimeError(f"Points feature step failed: {e}")

    # ============================================================
    # STEP 4 — Venue-specific W/D/L form (last 5 home / away)
    # ============================================================
    def venue_form_rolling(self):
        try:
            self.tracker.before("venue_form_rolling", self.df)

            df = self.df
            n0 = len(df)

            # ── Home-venue form ──
            hg = df[["date", "home_team", "home_goals", "away_goals"]].copy()
            hg.sort_values(["home_team", "date"], inplace=True)

            hg["h_win"]  = (hg["home_goals"] > hg["away_goals"]).astype(int)
            hg["h_draw"] = (hg["home_goals"] == hg["away_goals"]).astype(int)
            hg["h_loss"] = (hg["home_goals"] < hg["away_goals"]).astype(int)

            grp_h = hg.groupby("home_team")
            hg["hw5"] = grp_h["h_win"].rolling(5,  closed="left").sum().reset_index(0, drop=True)
            hg["hd5"] = grp_h["h_draw"].rolling(5, closed="left").sum().reset_index(0, drop=True)
            hg["hl5"] = grp_h["h_loss"].rolling(5, closed="left").sum().reset_index(0, drop=True)
            hg["hwr"] = hg["hw5"] / 5.0

            df = df.merge(
                hg[["date", "home_team", "hw5", "hd5", "hl5", "hwr"]],
                on=["date", "home_team"], how="left"
            )
            assert len(df) == n0, \
                f"Home venue merge changed row count {n0} -> {len(df)}."

            # ── Away-venue form ──
            # IMPORTANT: use ag (sorted by away_team+date), never reference hg here
            ag = df[["date", "away_team", "away_goals", "home_goals"]].copy()
            ag.sort_values(["away_team", "date"], inplace=True)

            ag["a_win"]  = (ag["away_goals"] > ag["home_goals"]).astype(int)
            ag["a_draw"] = (ag["away_goals"] == ag["home_goals"]).astype(int)
            ag["a_loss"] = (ag["away_goals"] < ag["home_goals"]).astype(int)

            grp_a = ag.groupby("away_team")
            ag["aw5"] = grp_a["a_win"].rolling(5,  closed="left").sum().reset_index(0, drop=True)
            ag["ad5"] = grp_a["a_draw"].rolling(5, closed="left").sum().reset_index(0, drop=True)
            ag["al5"] = grp_a["a_loss"].rolling(5, closed="left").sum().reset_index(0, drop=True)
            ag["awr"] = ag["aw5"] / 5.0

            df = df.merge(
                ag[["date", "away_team", "aw5", "ad5", "al5", "awr"]],
                on=["date", "away_team"], how="left"
            )
            assert len(df) == n0, \
                f"Away venue merge changed row count {n0} -> {len(df)}."

            df.rename(columns={
                "hw5": "home_team_home_wins_last5",
                "hd5": "home_team_home_draws_last5",
                "hl5": "home_team_home_losses_last5",
                "aw5": "away_team_away_wins_last5",
                "ad5": "away_team_away_draws_last5",
                "al5": "away_team_away_losses_last5",
            }, inplace=True)

            df["home_venue_advantage"] = df["hwr"] - df["awr"]
            df.drop(columns=["hwr", "awr"], inplace=True)

            # Drop rows where neither team has 5 venue games yet
            venue_cols = ["home_team_home_wins_last5", "away_team_away_wins_last5"]
            before = len(df)
            df.dropna(subset=venue_cols, inplace=True)
            logging.info(f"Venue NaN drop: {before - len(df)} rows "
                         f"(teams with <5 home or away games)")

            self.df = df
            self.tracker.after(self.df, "NaN rows = teams with <5 venue-specific games")
            return self

        except Exception as e:
            raise RuntimeError(f"Venue form feature step failed: {e}")

    # ============================================================
    # STEP 5 — Derived difference features
    # ============================================================
    def derived_features(self):
        try:
            self.tracker.before("derived_features", self.df)

            df = self.df

            df["points_diff_last5"]   = df["home_points_last5"]               - df["away_points_last5"]
            df["goal_diff_avg5"]      = df["home_goals_avg_last5"]            - df["away_goals_avg_last5"]
            df["xg_diff_avg5"]        = df["home_xg_avg_last5"]               - df["away_xg_avg_last5"]
            df["x_defense_diff"]      = df["away_goals_conceded_avg_last5"]   - df["home_goals_conceded_avg_last5"]
            df["ppda_diff_avg5"]      = df["home_ppda_avg_last5"]             - df["away_ppda_avg_last5"]
            df["deep_comp_diff_avg5"] = df["home_deep_completions_avg_last5"] - df["away_deep_completions_avg_last5"]
            df["venue_wins_diff"]     = df["home_team_home_wins_last5"]       - df["away_team_away_wins_last5"]
            df["home_advantage"]      = 1

            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)

            logging.info(f"Derived features added. Final shape: {df.shape}")
            logging.info(f"Final columns: {list(df.columns)}")

            self.df = df
            self.tracker.after(self.df, "pure computation — no rows should be lost")
            return self

        except Exception as e:
            raise RuntimeError(f"Derived feature step failed: {e}")

    # ============================================================
    # FINAL RUN
    # ============================================================
    def run(self):
        logging.info("\n========== PIPELINE START ==========")

        result = (
            self.basic_features()
                .rolling_features()
                .points_last5()
                .venue_form_rolling()
                .derived_features()
                .df
        )

        logging.info("\n========== PIPELINE COMPLETE ==========")
        self.tracker.print_audit()
        logging.info(f"Transformed data had {result.shape[0]} samples and {result.shape[1]} Columns")

        return result