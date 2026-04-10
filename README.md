# CIP Team 107 — ATP Tennis Match Prediction & Analysis

A school data-science project that explores historical ATP professional tennis
match data, engineers player-level features, and trains machine-learning models
to predict match outcomes.

---

## Project Structure

```
├── data/                   # Place ATP match CSV files here (see Data section)
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   └── 02_prediction.ipynb # Model training, evaluation & prediction
├── src/
│   ├── data_loader.py      # CSV loading, cleaning & synthetic data generator
│   ├── features.py         # Feature engineering (rolling stats, H2H, etc.)
│   ├── model.py            # Logistic Regression, Random Forest, XGBoost
│   └── visualizations.py  # Reusable plotting utilities
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Download real ATP match data

Download CSV files from [Jeff Sackmann's tennis_atp repository](https://github.com/JeffSackmann/tennis_atp)
(e.g. `atp_matches_2020.csv`, `atp_matches_2021.csv`, …) and place them in the
`data/` folder.

The notebooks also ship with a **synthetic data generator** that works without
any files, so you can run everything out of the box.

### 3. Launch Jupyter and open the notebooks

```bash
jupyter notebook
```

Open `notebooks/01_eda.ipynb` first for the exploratory analysis, then
`notebooks/02_prediction.ipynb` for model training and evaluation.

---

## Features Used for Prediction

| Feature | Description |
|---|---|
| `rank_diff` | Loser rank − Winner rank (positive = p1 is better ranked) |
| `p1_rank` / `p2_rank` | Current ATP ranking of each player |
| `p1_win_rate` / `p2_win_rate` | Rolling win rate over last 20 matches |
| `win_rate_diff` | Difference in rolling win rates |
| `p1_avg_rank` / `p2_avg_rank` | Average rank over last 20 matches |
| `avg_rank_diff` | Difference in average rank |
| `age_diff` | Age difference (p1 − p2) |
| `h2h_win_rate` | Head-to-head win rate of p1 vs p2 |
| `surface_*` | One-hot encoded surface (Hard / Clay / Grass) |
| `best_of_5` | Whether the match is best-of-5 sets |

---

## Models

Three binary classifiers are trained and compared using 5-fold
cross-validation:

- **Logistic Regression** — interpretable linear baseline
- **Random Forest** — ensemble of decision trees
- **XGBoost** — gradient-boosted trees (best performing)

---

## Team

CIP Team 107