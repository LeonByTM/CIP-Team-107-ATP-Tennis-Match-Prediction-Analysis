"""
data_loader.py
--------------
Utilities for loading and preprocessing ATP tennis match data.

Expected CSV format (Jeff Sackmann / tennis_atp style):
    tourney_id, tourney_name, surface, tourney_date,
    winner_id, winner_name, winner_rank, winner_age,
    loser_id,  loser_name,  loser_rank,  loser_age,
    score, round, best_of, minutes, ...
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

REQUIRED_COLS = [
    "tourney_date",
    "surface",
    "winner_name",
    "winner_rank",
    "winner_age",
    "loser_name",
    "loser_rank",
    "loser_age",
    "round",
    "best_of",
]


def load_matches(filepath: str) -> pd.DataFrame:
    """Load a single ATP match CSV file and apply basic cleaning."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Match data file not found: {filepath}")

    df = pd.read_csv(filepath, low_memory=False)
    df = _clean_matches(df)
    return df


def load_multiple(filepaths: list) -> pd.DataFrame:
    """Load and concatenate several ATP match CSV files."""
    frames = [load_matches(fp) for fp in filepaths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("tourney_date").reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning steps to a raw ATP match DataFrame."""
    # Parse date
    if "tourney_date" in df.columns:
        df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")

    # Surface standardisation
    if "surface" in df.columns:
        df["surface"] = df["surface"].str.strip().str.title()
        df["surface"] = df["surface"].replace({"Carpet": "Hard"})

    # Numeric coercion
    for col in ["winner_rank", "loser_rank", "winner_age", "loser_age", "minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing ranks with a high value (unranked)
    for col in ["winner_rank", "loser_rank"]:
        if col in df.columns:
            df[col] = df[col].fillna(1500)

    # Drop rows without a valid surface (required for modelling)
    if "surface" in df.columns:
        df = df.dropna(subset=["surface"])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demo / sample data (no real file required)
# ---------------------------------------------------------------------------

def generate_sample_data(n_matches: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic ATP-style match data for demonstration purposes.

    Each row represents one match.  The higher-ranked (lower rank number)
    player wins with probability ~0.65, surface adds a small random bias.
    """
    rng = np.random.default_rng(seed)

    surfaces = ["Hard", "Clay", "Grass"]
    rounds = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
    tournaments = [
        "Australian Open", "Roland Garros", "Wimbledon", "US Open",
        "Indian Wells", "Miami", "Monte Carlo", "Madrid",
        "Rome", "Canada", "Cincinnati", "Paris",
    ]

    rows = []
    player_pool = [f"Player_{i:03d}" for i in range(1, 151)]  # 150 unique players

    for _ in range(n_matches):
        # Draw two distinct players
        p1_idx, p2_idx = rng.choice(len(player_pool), size=2, replace=False)
        p1, p2 = player_pool[p1_idx], player_pool[p2_idx]

        # Ranks are correlated with index (lower idx → better player)
        p1_rank = int(rng.integers(max(1, p1_idx - 10), p1_idx + 20))
        p2_rank = int(rng.integers(max(1, p2_idx - 10), p2_idx + 20))

        surface = rng.choice(surfaces, p=[0.55, 0.30, 0.15])
        tourney = rng.choice(tournaments)
        year = int(rng.integers(2015, 2024))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 28))
        best_of = 5 if tourney in {"Australian Open", "Roland Garros", "Wimbledon", "US Open"} else 3
        rnd = rng.choice(rounds)

        # Better-ranked player wins with probability ~0.65
        rank_diff = p2_rank - p1_rank
        win_prob = 0.5 + 0.15 * np.tanh(rank_diff / 50)
        p1_wins = rng.random() < win_prob

        winner, loser = (p1, p2) if p1_wins else (p2, p1)
        w_rank, l_rank = (p1_rank, p2_rank) if p1_wins else (p2_rank, p1_rank)
        w_age = 20 + (p1_idx if p1_wins else p2_idx) % 15 + rng.integers(0, 3)
        l_age = 20 + (p2_idx if p1_wins else p1_idx) % 15 + rng.integers(0, 3)

        rows.append({
            "tourney_date": f"{year}{month:02d}{day:02d}",
            "tourney_name": tourney,
            "surface": surface,
            "round": rnd,
            "best_of": best_of,
            "winner_name": winner,
            "winner_rank": w_rank,
            "winner_age": int(w_age),
            "loser_name": loser,
            "loser_rank": l_rank,
            "loser_age": int(l_age),
            "minutes": int(rng.integers(60, 240)),
        })

    df = pd.DataFrame(rows)
    return _clean_matches(df)
