# FIFA World Cup 2026 Prediction Project

This repository contains a full modeling and simulation workflow for forecasting the **2026 FIFA World Cup** using a mix of:

- probabilistic football models,
- historical international match results,
- FC 26 player ratings,
- tournament simulation logic,
- and export scripts for tables, SVG visuals, and notebook-based reporting.

The project was built for transparent, explainable tournament forecasting rather than just producing a single winner pick. It is designed to answer questions like:

- Which teams are strongest before the tournament?
- How much do squad ratings matter compared with form and rankings?
- What is the most likely tournament path?
- Which teams are most likely to win their group, qualify for the knockout stage, or become world champions?

## Project Overview

The repository combines several components into one pipeline:

1. **Feature engineering**
   Historical team form, Elo-like ratings, FC 26 squad strength, rankings, weather proxies, and population features are merged into match-level features.

2. **Match prediction**
   Three separate models are trained for international football prediction:
   - `Poisson`
   - `Logistic Regression`
   - `XGBoost`

3. **Tournament simulation**
   The 48-team 2026 World Cup format is simulated with:
   - 12 groups of 4
   - best third-placed teams advancing
   - Round of 32 through the Final

4. **Exports for analysis**
   The project produces:
   - champion probability tables,
   - group-stage probability tables,
   - deterministic “most likely path” brackets,
   - SVG graphics,
   - and notebook-ready summary outputs.

## Repository Structure

```text
.
├── worldcup_models.py
├── simulate_worldcup_2026.py
├── build_worldcup_likely_paths.py
├── build_worldcup_bracket_visual.py
├── build_worldcup_report_notebook.py
├── build_xgb_group_probabilities.py
├── build_xgb_group_probabilities_svg.py
├── clean_country_population.py
├── data/
│   ├── group_draw_2026.csv
│   └── country_population_cleaned.csv
└── worldcup_2026_outputs/
    └── .gitkeep
```

## Main Files

### Core Modeling

- [worldcup_models.py](/Users/niklasrotter/Documents/Videoprojekte/SportsBetting/worldcup_models.py)
  Builds the full modeling dataset, engineers features, trains the models, and exposes helper functions for match prediction.

- [simulate_worldcup_2026.py](/Users/niklasrotter/Documents/Videoprojekte/SportsBetting/simulate_worldcup_2026.py)
  Runs Monte Carlo World Cup simulations for `poisson`, `logistic`, and `xgb`.

### Reporting and Visual Output

- [build_worldcup_likely_paths.py](/Users/niklasrotter/Documents/Videoprojekte/SportsBetting/build_worldcup_likely_paths.py)
  Creates deterministic “most likely” tournament paths for each model.

- [build_worldcup_bracket_visual.py](/Users/niklasrotter/Documents/Videoprojekte/SportsBetting/build_worldcup_bracket_visual.py)
  Generates bracket and group-stage visuals for a featured run.

- [build_worldcup_report_notebook.py](/Users/niklasrotter/Documents/Videoprojekte/SportsBetting/build_worldcup_report_notebook.py)
  Builds a notebook report that summarizes the simulation outputs.

- [build_xgb_group_probabilities.py](/Users/niklasrotter/Documents/Videoprojekte/SportsBetting/build_xgb_group_probabilities.py)
  Computes group-stage placement and qualification probabilities for the XGBoost model.

- [build_xgb_group_probabilities_svg.py](/Users/niklasrotter/Documents/Videoprojekte/SportsBetting/build_xgb_group_probabilities_svg.py)
  Converts grouped XGBoost group probabilities into a presentation-ready SVG table.

### Data Cleaning

- [clean_country_population.py](/Users/niklasrotter/Documents/Videoprojekte/SportsBetting/clean_country_population.py)
  Cleans the raw population CSV and exports a normalized file used in the model.

## Data Sources

The Transfermarkt dataset, distributed on GitHub
(https://github.com/dcaribou/transfermarkt-datasets)
and the FC26 player ratings, published on kaggle (https://www.kaggle.com/datasets/rovnez/fc-26-fifa-26-player-data)
were used to train the models. Weather data and population sizes were
estimated.

- `archive/games.csv`
  Historical match results used for form and Elo-like features (Transfermarkt).

- `archive/national_teams.csv`
  Team-level metadata such as FIFA ranking and squad-related context (Transfermarkt).

- `FC26_20250921.csv`
  FC 26 player ratings used to create squad-strength features (Kaggle).

### Versioned with the repository

- `data/group_draw_2026.csv`
  The current 2026 group draw used by the simulation scripts.

- `data/country_population_cleaned.csv`
  Cleaned population input used for the contextual country features.


## Features Used by the Models

The feature set is a blend of football-specific and contextual variables.

### Team strength and squad quality

- FC 26 squad `overall`, `potential`, `attack`, `defense`, `passing`, `playmaker`, goalkeeper ratings
- top 11 and top 23 squad averages
- derived squad balance features

### Historical performance

- rolling form points over the last 5 matches
- rolling goal difference over the last 5 matches
- recent goals scored and conceded
- Elo-style team rating

### Team metadata

- FIFA ranking
- market value features
- squad age and composition

### Contextual features

- weather proxies such as average temperature and indoor/roof adjustments
- country population features

Most team-quality features are used as **home-away differences**, because the models predict from the perspective of one team against another.

## Models

### 1. Poisson

The Poisson setup models expected goals for each team and converts those goal expectations into match outcome probabilities.

This is useful because:

- it is interpretable,
- it is football-native,
- and it gives expected scoreline structure rather than only class labels.

### 2. Logistic Regression

The logistic model predicts:

- home win,
- draw,
- away win

It acts as a strong linear baseline and helps compare how much nonlinear modeling is actually adding.

### 3. XGBoost

The XGBoost classifier is the most flexible model in the repository and usually captures the most complex interactions between:

- squad quality,
- rankings,
- form,
- and contextual features.

This is also the model used for many of the group-stage and bracket probability exports.

## 2026 Tournament Logic

The tournament simulation follows the **48-team World Cup format**:

- 12 groups of 4
- top 2 teams from each group advance automatically
- the 8 best third-placed teams also advance
- knockout stage starts in the Round of 32

Important note:

The Round of 32 is **not** simply “group winner vs runner-up.”  
It uses a structured mapping involving group winners, runners-up, and qualified third-placed teams, which is why some first knockout matchups may look unusual at first glance.

## Setup

### Python Version

Recommended:

- Python `3.11+`

### Suggested Dependencies

Install the core libraries used in the scripts:

```bash
pip install numpy pandas scipy scikit-learn xgboost matplotlib seaborn nbformat
```

Depending on how you want to use the project, you may also want:

```bash
pip install jupyter ipython
```

## Typical Workflow

### 1. Train models and run tournament simulations

```bash
python simulate_worldcup_2026.py --sims 10000 --methods poisson logistic xgb
```

This produces outputs such as:

- `champions_poisson.csv`
- `champions_logistic.csv`
- `champions_xgb.csv`
- `stage_probs_poisson.csv`
- `stage_probs_logistic.csv`
- `stage_probs_xgb.csv`

### 2. Build the deterministic “most likely path”

```bash
python build_worldcup_likely_paths.py
```

This produces:

- likely-path bracket tables
- likely-path group standings
- SVG bracket exports

### 3. Build XGBoost group-stage probabilities

```bash
python build_xgb_group_probabilities.py
python build_xgb_group_probabilities_svg.py
```

This produces:

- `group_probs_xgb_by_group.csv`
- `group_probs_xgb_by_group.svg`

### 4. Build the notebook report

```bash
python build_worldcup_report_notebook.py
```

Then open the generated notebook from the project root if you want a report export.

## Output Files

Some of the most important generated outputs are:

- `worldcup_2026_outputs/champions_xgb.csv`
  Champion probabilities from the XGBoost model.

- `worldcup_2026_outputs/stage_probs_xgb.csv`
  Probability of reaching the semifinal, final, and winning the tournament.

- `worldcup_2026_outputs/group_probs_xgb_by_group.csv`
  Group win, runner-up, third-place, fourth-place, and qualification probabilities.

- `worldcup_2026_outputs/likely_xgb.svg`
  Deterministic most-likely bracket path for XGBoost.

- `worldcup_2026_outputs/group_probs_xgb_by_group.svg`
  SVG summary table for all groups using XGBoost probabilities.

- `worldcup_2026_outputs/xgb_feature_importances_named.csv`
  Feature importance export for XGBoost with readable feature names.

## Reproducibility

Most scripts use fixed random seeds to make outputs more stable and reproducible.  
That said, changes to:

- the group draw,
- input data files,
- feature engineering,
- or model hyperparameters

will naturally change the tournament forecasts.

## Assumptions and Limitations

This repository is built to be practical and explainable, but there are still some important limitations:

- FC 26 ratings are modern ratings and therefore best interpreted as a **2026 pre-tournament strength prior**, not as historically time-consistent ratings for older tournaments.
- Weather data is partly proxy-based rather than match-specific meteorological data.
- Population is a weak contextual feature and should not be overinterpreted causally.
- The “most likely path” is a deterministic path built from local match probabilities, not a full global optimization over every possible bracket tree.
- Some optional data files are loaded from local paths and may not be present in every clone of the repository.

