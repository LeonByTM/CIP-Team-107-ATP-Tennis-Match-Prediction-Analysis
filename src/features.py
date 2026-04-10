"""
features.py
-----------
Feature engineering for ATP match prediction.

Given a DataFrame of historical matches (one row per match, winner/loser
format), this module computes player-level rolling statistics and builds
a modelling DataFrame where each row represents a match-up with features
derived from both players' recent history.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rolling player statistics
# ---------------------------------------------------------------------------

def compute_player_stats(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute per-player rolling statistics up to (but not including) each match.

    Returns a copy of *df* with additional columns:
        winner_win_rate_<window>, winner_avg_rank_<window>
        loser_win_rate_<window>,  loser_avg_rank_<window>
    """
    df = df.copy().sort_values("tourney_date").reset_index(drop=True)

    # Build a per-player history incrementally
    player_wins: dict = {}    # name -> deque of 1/0
    player_ranks: dict = {}   # name -> deque of ranks

    win_rates_w, win_rates_l = [], []
    avg_ranks_w, avg_ranks_l = [], []

    for _, row in df.iterrows():
        w, l = row["winner_name"], row["loser_name"]
        wr, lr = row["winner_rank"], row["loser_rank"]

        # Stats BEFORE this match (use what we've accumulated so far)
        w_hist = player_wins.get(w, [])
        l_hist = player_wins.get(l, [])
        w_rank_hist = player_ranks.get(w, [])
        l_rank_hist = player_ranks.get(l, [])

        win_rates_w.append(np.mean(w_hist[-window:]) if w_hist else 0.5)
        win_rates_l.append(np.mean(l_hist[-window:]) if l_hist else 0.5)
        avg_ranks_w.append(np.mean(w_rank_hist[-window:]) if w_rank_hist else wr)
        avg_ranks_l.append(np.mean(l_rank_hist[-window:]) if l_rank_hist else lr)

        # Update history
        player_wins.setdefault(w, []).append(1)
        player_wins.setdefault(l, []).append(0)
        player_ranks.setdefault(w, []).append(wr)
        player_ranks.setdefault(l, []).append(lr)

    suffix = f"_{window}"
    df[f"winner_win_rate{suffix}"] = win_rates_w
    df[f"loser_win_rate{suffix}"] = win_rates_l
    df[f"winner_avg_rank{suffix}"] = avg_ranks_w
    df[f"loser_avg_rank{suffix}"] = avg_ranks_l

    return df


# ---------------------------------------------------------------------------
# Head-to-head features
# ---------------------------------------------------------------------------

def compute_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column ``winner_h2h_win_rate`` = wins / (wins + losses) of the
    eventual winner against the eventual loser *before* this match.
    """
    df = df.copy().sort_values("tourney_date").reset_index(drop=True)

    h2h: dict = {}  # (player_a, player_b) -> [wins_a, wins_b]

    rates = []
    for _, row in df.iterrows():
        w, l = row["winner_name"], row["loser_name"]
        key = tuple(sorted([w, l]))
        rec = h2h.get(key, [0, 0])

        total = rec[0] + rec[1]
        if total == 0:
            rates.append(0.5)
        else:
            # Fraction of wins for the eventual winner
            w_wins = rec[0] if key[0] == w else rec[1]
            rates.append(w_wins / total)

        # Update
        entry = h2h.setdefault(key, [0, 0])
        if key[0] == w:
            entry[0] += 1
        else:
            entry[1] += 1

    df["winner_h2h_win_rate"] = rates
    return df


# ---------------------------------------------------------------------------
# Build modelling dataset
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Transform raw match data into a feature matrix for binary classification.

    Target: 1 = player_1 wins (player with the lower rank in the row),
            0 = player_2 wins.

    We randomly assign winner/loser to player_1/player_2 so the model does
    not simply learn "winner always appears in column 1".
    """
    df = compute_player_stats(df, window=window)
    df = compute_h2h(df)

    rng = np.random.default_rng(0)
    swap = rng.random(len(df)) < 0.5  # randomly swap roles

    feature_rows = []
    targets = []

    for i, row in df.iterrows():
        flipped = swap[i]

        if not flipped:
            p1_rank = row["winner_rank"]
            p2_rank = row["loser_rank"]
            p1_win_rate = row[f"winner_win_rate_{window}"]
            p2_win_rate = row[f"loser_win_rate_{window}"]
            p1_avg_rank = row[f"winner_avg_rank_{window}"]
            p2_avg_rank = row[f"loser_avg_rank_{window}"]
            p1_age = row["winner_age"]
            p2_age = row["loser_age"]
            h2h_rate = row["winner_h2h_win_rate"]
            target = 1
        else:
            p1_rank = row["loser_rank"]
            p2_rank = row["winner_rank"]
            p1_win_rate = row[f"loser_win_rate_{window}"]
            p2_win_rate = row[f"winner_win_rate_{window}"]
            p1_avg_rank = row[f"loser_avg_rank_{window}"]
            p2_avg_rank = row[f"winner_avg_rank_{window}"]
            p1_age = row["loser_age"]
            p2_age = row["winner_age"]
            h2h_rate = 1.0 - row["winner_h2h_win_rate"]
            target = 0

        surface = row.get("surface", "Hard")
        best_of = row.get("best_of", 3)

        feature_rows.append({
            "rank_diff": p2_rank - p1_rank,          # positive → p1 is better
            "p1_rank": p1_rank,
            "p2_rank": p2_rank,
            "p1_win_rate": p1_win_rate,
            "p2_win_rate": p2_win_rate,
            "win_rate_diff": p1_win_rate - p2_win_rate,
            "p1_avg_rank": p1_avg_rank,
            "p2_avg_rank": p2_avg_rank,
            "avg_rank_diff": p2_avg_rank - p1_avg_rank,
            "age_diff": p1_age - p2_age,
            "h2h_win_rate": h2h_rate,
            "surface_clay": int(surface == "Clay"),
            "surface_grass": int(surface == "Grass"),
            "surface_hard": int(surface == "Hard"),
            "best_of_5": int(best_of == 5),
        })
        targets.append(target)

    X = pd.DataFrame(feature_rows, index=df.index)
    y = pd.Series(targets, index=df.index, name="target")
    return X, y


FEATURE_NAMES = [
    "rank_diff",
    "p1_rank",
    "p2_rank",
    "p1_win_rate",
    "p2_win_rate",
    "win_rate_diff",
    "p1_avg_rank",
    "p2_avg_rank",
    "avg_rank_diff",
    "age_diff",
    "h2h_win_rate",
    "surface_clay",
    "surface_grass",
    "surface_hard",
    "best_of_5",
]
