from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_GAMES = BASE_DIR / "archive" / "games.csv"
DATA_TEAMS = BASE_DIR / "archive" / "national_teams.csv"
DATA_PLAYERS = BASE_DIR / "FC26_20250921.csv"
WEATHER_HISTORY_FILENAME = "WorldCupTemps - okay, it would be best to also have historical te....csv"
WEATHER_2026_FILENAME = "WorldCup2026Temps - I am working on a machine learning project for pr....csv"
POPULATION_CLEAN_FILENAME = "country_population_cleaned.csv"

FC26_NUMERIC_COLUMNS = [
    "overall",
    "potential",
    "value_eur",
    "age",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
]

TEAM_META_COLUMNS = [
    "fifa_ranking",
    "total_market_value",
    "squad_size",
    "average_age",
    "foreigners_percentage",
]

RESULT_LABELS = {0: "home_win", 1: "draw", 2: "away_win"}

TOURNAMENT_HOSTS = {
    ("FIWC", 2014): "Brazil",
    ("FIWC", 2018): "Russia",
    ("FIWC", 2022): "Qatar",
    ("EURO", 2016): "France",
    ("EURO", 2024): "Germany",
    ("COPA", 2015): "Chile",
    ("COPA", 2016): "United States",
    ("COPA", 2019): "Brazil",
    ("COPA", 2021): "Brazil",
    ("COPA", 2024): "United States",
    ("AFCN", 2015): "Equatorial Guinea",
    ("AFCN", 2017): "Gabon",
    ("AFCN", 2019): "Egypt",
    ("AFCN", 2021): "Cameroon",
    ("AFCN", 2023): "Ivory Coast",
    ("AFAC", 2015): "Australia",
    ("AFAC", 2019): "United Arab Emirates",
    ("AFAC", 2023): "Qatar",
}


def normalize_team_name(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_log1p(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    value = float(value)
    return float(np.log1p(max(value, 0.0)))


def canonicalize_team(value: object, alias_map: Dict[str, str]) -> Optional[str]:
    normalized = normalize_team_name(value)
    if not normalized:
        return None
    if normalized in alias_map:
        return alias_map[normalized]
    return str(value).strip()


def load_sources(base_dir: Path = BASE_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = pd.read_csv(DATA_GAMES)
    teams = pd.read_csv(DATA_TEAMS)
    players = pd.read_csv(DATA_PLAYERS, low_memory=False)
    return games, teams, players


def resolve_optional_data_path(filename: str, extra_dirs: Sequence[Path] = ()) -> Optional[Path]:
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR / "worldcup_2026_outputs" / filename,
        Path.home() / "Downloads" / filename,
    ]
    candidates.extend(Path(directory) / filename for directory in extra_dirs)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_indoor_flag(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if not text:
        return np.nan
    if text.startswith("yes") or "retractable" in text or "indoor" in text or "ac" in text or "enclosed" in text:
        return 1.0
    if text.startswith("no"):
        return 0.0
    return np.nan


def load_weather_sources() -> Tuple[pd.DataFrame, pd.DataFrame]:
    history_path = resolve_optional_data_path(WEATHER_HISTORY_FILENAME)
    future_path = resolve_optional_data_path(WEATHER_2026_FILENAME)

    history = pd.read_csv(history_path) if history_path is not None else pd.DataFrame()
    future = pd.read_csv(future_path) if future_path is not None else pd.DataFrame()
    return history, future


def build_weather_context() -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    history, future = load_weather_sources()

    history_lookup: Dict[int, Dict[str, float]] = {}
    if not history.empty and "Tournament" in history.columns:
        history = history.copy()
        history["tournament_year"] = history["Tournament"].astype(str).str.extract(r"(\d{4})")[0]
        history["tournament_year"] = pd.to_numeric(history["tournament_year"], errors="coerce")
        history["weather_indoor_ac"] = history["Indoor / AC?"].map(parse_indoor_flag)
        history["avg_temp_c"] = pd.to_numeric(history["Avg Temp (°C)"], errors="coerce")
        history["max_temp_c"] = pd.to_numeric(history["Max Temp Recorded (°C)"], errors="coerce")

        for row in history.dropna(subset=["tournament_year"]).itertuples(index=False):
            year = int(row.tournament_year)
            avg_temp = float(row.avg_temp_c) if not pd.isna(row.avg_temp_c) else np.nan
            max_temp = float(row.max_temp_c) if not pd.isna(row.max_temp_c) else np.nan
            indoor = float(row.weather_indoor_ac) if not pd.isna(row.weather_indoor_ac) else np.nan
            history_lookup[year] = {
                "weather_avg_temp_c": avg_temp,
                "weather_max_temp_c": max_temp,
                "weather_indoor_ac": indoor,
                "weather_adjusted_temp_c": avg_temp - (4.0 * indoor) if not pd.isna(avg_temp) and not pd.isna(indoor) else np.nan,
                "weather_available": 1.0,
            }

    future_context: Dict[str, float] = {
        "weather_avg_temp_c": np.nan,
        "weather_max_temp_c": np.nan,
        "weather_indoor_ac": np.nan,
        "weather_adjusted_temp_c": np.nan,
        "weather_available": 0.0,
    }
    if not future.empty:
        future = future.copy()
        future["weather_indoor_ac"] = future["Indoor / Retractable Roof?"].map(parse_indoor_flag)
        future["forecast_temp_c"] = pd.to_numeric(
            future["2026 Summer Forecast Estimate (°C)"], errors="coerce"
        )
        avg_temp = float(future["forecast_temp_c"].mean()) if future["forecast_temp_c"].notna().any() else np.nan
        max_temp = float(future["forecast_temp_c"].max()) if future["forecast_temp_c"].notna().any() else np.nan
        indoor = float(future["weather_indoor_ac"].mean()) if future["weather_indoor_ac"].notna().any() else np.nan
        future_context = {
            "weather_avg_temp_c": avg_temp,
            "weather_max_temp_c": max_temp,
            "weather_indoor_ac": indoor,
            "weather_adjusted_temp_c": avg_temp - (4.0 * indoor) if not pd.isna(avg_temp) and not pd.isna(indoor) else np.nan,
            "weather_available": 1.0,
    }
    return history_lookup, future_context


def load_population_source() -> pd.DataFrame:
    population_path = resolve_optional_data_path(POPULATION_CLEAN_FILENAME)
    if population_path is None:
        return pd.DataFrame()
    return pd.read_csv(population_path)


def build_population_features(alias_map: Dict[str, str]) -> pd.DataFrame:
    population = load_population_source()
    if population.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "country_population",
                "country_population_millions",
                "country_population_log1p",
                "country_population_share_world",
            ]
        )

    population = population.copy()
    source_location = "location_clean" if "location_clean" in population.columns else "Location"
    population["team"] = population[source_location].map(lambda x: canonicalize_team(x, alias_map))
    population = population.dropna(subset=["team"]).copy()

    if "population" in population.columns:
        population["country_population"] = pd.to_numeric(population["population"], errors="coerce")
    else:
        population["country_population"] = pd.to_numeric(population.get("Population"), errors="coerce")
    population["country_population_millions"] = population["country_population"] / 1_000_000.0
    population["country_population_log1p"] = np.log1p(population["country_population"])
    if "world_share_pct" in population.columns:
        population["country_population_share_world"] = pd.to_numeric(
            population["world_share_pct"], errors="coerce"
        )
    else:
        population["country_population_share_world"] = np.nan

    return population.groupby("team", as_index=False).agg(
        country_population=("country_population", "max"),
        country_population_millions=("country_population_millions", "max"),
        country_population_log1p=("country_population_log1p", "max"),
        country_population_share_world=("country_population_share_world", "max"),
    )


def build_alias_map(games: pd.DataFrame, teams: pd.DataFrame) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}

    game_team_names = sorted(
        set(games["home_club_name"].dropna().astype(str))
        | set(games["away_club_name"].dropna().astype(str))
    )
    for team_name in game_team_names:
        alias_map.setdefault(normalize_team_name(team_name), team_name)

    for _, row in teams.iterrows():
        canonical_name = str(row["name"]).strip()
        for candidate in [row.get("name"), row.get("country_name")]:
            norm = normalize_team_name(candidate)
            if norm and norm not in alias_map:
                alias_map[norm] = canonical_name

    manual_aliases = {
        "usa": "United States",
        "bosnia and herzegovina": "Bosnia-Herzegovina",
        "bosnia herzegovina": "Bosnia-Herzegovina",
        "czechia": "Czech Republic",
        "ir iran": "Iran",
        "iran": "Iran",
        "korea republic": "South Korea",
        "south korea": "South Korea",
        "china pr": "China",
        "republic of ireland": "Republic of Ireland",
        "ireland": "Republic of Ireland",
        "hongkong": "Hong Kong",
        "cabo verde": "Cape Verde",
        "cape verde": "Cape Verde",
        "cote divoire": "Ivory Coast",
        "cote d ivoire": "Ivory Coast",
        "ivory coast": "Ivory Coast",
        "congo dr": "Democratic Republic of the Congo",
        "democratic republic of the congo": "Democratic Republic of the Congo",
        "curaçao": "Curaçao",
        "curacao": "Curaçao",
        "turkey": "Turkiye",
        "turkiye": "Turkiye",
    }
    alias_map.update(manual_aliases)
    return alias_map


def build_team_metadata(teams: pd.DataFrame, alias_map: Dict[str, str]) -> pd.DataFrame:
    meta = teams.copy()
    meta["team"] = meta["name"].map(lambda x: canonicalize_team(x, alias_map))
    meta["country_team"] = meta["country_name"].map(lambda x: canonicalize_team(x, alias_map))
    meta["team"] = meta["team"].fillna(meta["country_team"])
    meta = meta.dropna(subset=["team"]).copy()

    meta = meta.groupby("team", as_index=False).agg(
        fifa_ranking=("fifa_ranking", "min"),
        total_market_value=("total_market_value", "max"),
        squad_size=("squad_size", "max"),
        average_age=("average_age", "mean"),
        foreigners_percentage=("foreigners_percentage", "mean"),
        confederation=("confederation", "first"),
    )

    meta["total_market_value_log1p"] = meta["total_market_value"].map(safe_log1p)
    meta["fifa_ranking_inverse"] = 1.0 / meta["fifa_ranking"].clip(lower=1)
    return meta


def _parse_gk_value(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).split("+")[0].strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def build_fc26_team_features(
    players: pd.DataFrame, alias_map: Dict[str, str], top_n: int = 23
) -> pd.DataFrame:
    data = players.copy()
    data["team"] = data["nationality_name"].map(lambda x: canonicalize_team(x, alias_map))
    data = data.dropna(subset=["team"]).copy()

    for column in FC26_NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["gk_rating"] = data["gk"].map(_parse_gk_value)
    data["attack_index"] = data[["shooting", "dribbling", "passing"]].mean(axis=1)
    data["defense_index"] = data[["defending", "physic"]].mean(axis=1)
    data["playmaker_index"] = data[["passing", "dribbling", "overall"]].mean(axis=1)

    feature_columns = FC26_NUMERIC_COLUMNS + [
        "gk_rating",
        "attack_index",
        "defense_index",
        "playmaker_index",
    ]

    team_rows: List[Dict[str, float]] = []
    for team, team_players in data.groupby("team"):
        team_players = team_players.sort_values("overall", ascending=False).head(top_n)
        if team_players.empty:
            continue

        row: Dict[str, float] = {"team": team, "fc26_squad_size": float(len(team_players))}
        row["fc26_top11_overall_mean"] = float(team_players.head(11)["overall"].mean())
        row["fc26_top23_overall_mean"] = float(team_players["overall"].mean())
        row["fc26_overall_max"] = float(team_players["overall"].max())
        row["fc26_overall_std"] = float(team_players["overall"].std(ddof=0))
        row["fc26_depth_80"] = float((team_players["overall"] >= 80).sum())
        row["fc26_depth_85"] = float((team_players["overall"] >= 85).sum())
        row["fc26_value_log1p_sum"] = float(np.log1p(team_players["value_eur"].fillna(0)).sum())
        row["fc26_age_mean"] = float(team_players["age"].mean())
        row["fc26_pace_mean"] = float(team_players["pace"].mean())
        row["fc26_shooting_mean"] = float(team_players["shooting"].mean())
        row["fc26_passing_mean"] = float(team_players["passing"].mean())
        row["fc26_dribbling_mean"] = float(team_players["dribbling"].mean())
        row["fc26_defending_mean"] = float(team_players["defending"].mean())
        row["fc26_physic_mean"] = float(team_players["physic"].mean())
        row["fc26_attack_index_mean"] = float(team_players["attack_index"].mean())
        row["fc26_defense_index_mean"] = float(team_players["defense_index"].mean())
        row["fc26_playmaker_index_mean"] = float(team_players["playmaker_index"].mean())
        row["fc26_gk_rating_max"] = float(team_players["gk_rating"].max())
        row["fc26_gk_rating_mean"] = float(team_players["gk_rating"].mean())

        for column in feature_columns:
            row[f"fc26_{column}_mean"] = float(team_players[column].mean())

        team_rows.append(row)

    team_features = pd.DataFrame(team_rows)
    team_features["fc26_overall_balance"] = (
        0.45 * team_features["fc26_top11_overall_mean"]
        + 0.25 * team_features["fc26_attack_index_mean"]
        + 0.20 * team_features["fc26_defense_index_mean"]
        + 0.10 * team_features["fc26_gk_rating_max"].fillna(0)
    )
    return team_features


def build_rolling_form_features(matches: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, deque]]:
    ratings: Dict[str, float] = defaultdict(lambda: 1500.0)
    form: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
    rows: List[Dict[str, float]] = []

    for row in matches.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        home_goals = int(row.home_club_goals)
        away_goals = int(row.away_club_goals)

        home_points = [entry[0] for entry in form[home]]
        away_points = [entry[0] for entry in form[away]]
        home_gd = [entry[1] for entry in form[home]]
        away_gd = [entry[1] for entry in form[away]]

        home_elo = ratings[home]
        away_elo = ratings[away]
        elo_diff = home_elo - away_elo

        rows.append(
            {
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": elo_diff,
                "home_form_points_5": float(np.mean(home_points)) if home_points else 1.0,
                "away_form_points_5": float(np.mean(away_points)) if away_points else 1.0,
                "home_form_gd_5": float(np.mean(home_gd)) if home_gd else 0.0,
                "away_form_gd_5": float(np.mean(away_gd)) if away_gd else 0.0,
                "home_recent_goals_for_5": float(
                    np.mean([entry[2] for entry in form[home]])
                )
                if form[home]
                else 1.0,
                "away_recent_goals_for_5": float(
                    np.mean([entry[2] for entry in form[away]])
                )
                if form[away]
                else 1.0,
                "home_recent_goals_against_5": float(
                    np.mean([entry[3] for entry in form[home]])
                )
                if form[home]
                else 1.0,
                "away_recent_goals_against_5": float(
                    np.mean([entry[3] for entry in form[away]])
                )
                if form[away]
                else 1.0,
            }
        )

        if home_goals > away_goals:
            home_score, away_score = 1.0, 0.0
            home_points_val, away_points_val = 3.0, 0.0
        elif home_goals < away_goals:
            home_score, away_score = 0.0, 1.0
            home_points_val, away_points_val = 0.0, 3.0
        else:
            home_score = away_score = 0.5
            home_points_val = away_points_val = 1.0

        goal_margin = abs(home_goals - away_goals)
        elo_expected = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))
        margin_multiplier = np.log1p(goal_margin) + 1.0
        k_factor = 20.0 * margin_multiplier
        ratings[home] += k_factor * (home_score - elo_expected)
        ratings[away] += k_factor * (away_score - (1.0 - elo_expected))

        form[home].append(
            (
                home_points_val,
                float(home_goals - away_goals),
                float(home_goals),
                float(away_goals),
            )
        )
        form[away].append(
            (
                away_points_val,
                float(away_goals - home_goals),
                float(away_goals),
                float(home_goals),
            )
        )

    return pd.DataFrame(rows), ratings, form


def summarize_current_state(
    ratings: Dict[str, float], form: Dict[str, deque]
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    all_teams = sorted(set(ratings.keys()) | set(form.keys()))
    for team in all_teams:
        team_form = list(form.get(team, []))
        rows.append(
            {
                "team": team,
                "current_elo": float(ratings.get(team, 1500.0)),
                "current_form_points_5": float(np.mean([entry[0] for entry in team_form])) if team_form else 1.0,
                "current_form_gd_5": float(np.mean([entry[1] for entry in team_form])) if team_form else 0.0,
                "current_recent_goals_for_5": float(np.mean([entry[2] for entry in team_form])) if team_form else 1.0,
                "current_recent_goals_against_5": float(np.mean([entry[3] for entry in team_form])) if team_form else 1.0,
            }
        )
    return pd.DataFrame(rows)


def build_match_dataset(
    games: pd.DataFrame,
    team_features: pd.DataFrame,
    alias_map: Dict[str, str],
) -> pd.DataFrame:
    data = games.copy()
    data = data[data["competition_type"].eq("national_team_competition")].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["home_team"] = data["home_club_name"].map(lambda x: canonicalize_team(x, alias_map))
    data["away_team"] = data["away_club_name"].map(lambda x: canonicalize_team(x, alias_map))
    data = data.dropna(subset=["date", "home_team", "away_team"]).copy()
    data = data.sort_values("date").reset_index(drop=True)

    weather_lookup, _ = build_weather_context()
    for column in [
        "weather_available",
        "weather_avg_temp_c",
        "weather_max_temp_c",
        "weather_indoor_ac",
        "weather_adjusted_temp_c",
    ]:
        data[column] = np.nan
    data["weather_available"] = 0.0
    data["tournament_year"] = pd.to_numeric(data["date"].dt.year, errors="coerce")
    fiwc_mask = data["competition_id"].eq("FIWC")
    if weather_lookup:
        avg_map = {year: values["weather_avg_temp_c"] for year, values in weather_lookup.items()}
        max_map = {year: values["weather_max_temp_c"] for year, values in weather_lookup.items()}
        indoor_map = {year: values["weather_indoor_ac"] for year, values in weather_lookup.items()}
        adjusted_map = {year: values["weather_adjusted_temp_c"] for year, values in weather_lookup.items()}
        data.loc[fiwc_mask, "weather_avg_temp_c"] = data.loc[fiwc_mask, "tournament_year"].map(avg_map)
        data.loc[fiwc_mask, "weather_max_temp_c"] = data.loc[fiwc_mask, "tournament_year"].map(max_map)
        data.loc[fiwc_mask, "weather_indoor_ac"] = data.loc[fiwc_mask, "tournament_year"].map(indoor_map)
        data.loc[fiwc_mask, "weather_adjusted_temp_c"] = data.loc[fiwc_mask, "tournament_year"].map(adjusted_map)
        data.loc[fiwc_mask, "weather_available"] = data.loc[fiwc_mask, "weather_avg_temp_c"].notna().astype(float)

    rolling, ratings, form = build_rolling_form_features(data)
    data = pd.concat([data, rolling], axis=1)

    home_features = team_features.rename(columns={col: f"home_{col}" for col in team_features.columns if col != "team"})
    away_features = team_features.rename(columns={col: f"away_{col}" for col in team_features.columns if col != "team"})
    home_features = home_features.rename(columns={"team": "home_team"})
    away_features = away_features.rename(columns={"team": "away_team"})

    data = data.merge(home_features, on="home_team", how="left").merge(away_features, on="away_team", how="left")

    static_cols = [col for col in team_features.select_dtypes(include=[np.number]).columns if col != "team"]
    for col in static_cols:
        data[f"{col}_diff"] = data[f"home_{col}"] - data[f"away_{col}"]

    normalized_hosts = {
        (comp, int(season)): canonicalize_team(host, alias_map)
        for (comp, season), host in TOURNAMENT_HOSTS.items()
        if not pd.isna(season)
    }

    def is_home_adv(row):
        comp = row.get("competition_id")
        year = row.get("tournament_year")
        home = row.get("home_team")
        if pd.isna(year):
            return 0.0
        host = normalized_hosts.get((comp, int(year)))
        if host and home == host:
            return 1.0
        if comp == "EURO" and int(year) == 2020:
            euro_hosts = {"Italy", "England", "Germany", "Azerbaijan", "Russia", "Spain", "Romania", "Hungary", "Denmark", "Netherlands", "Scotland"}
            norm_euro_hosts = {canonicalize_team(h, alias_map) for h in euro_hosts}
            if home in norm_euro_hosts:
                return 1.0
        return 0.0

    data["home_advantage"] = data.apply(is_home_adv, axis=1)
    data["result"] = np.select(
        [data["home_club_goals"] > data["away_club_goals"], data["home_club_goals"] == data["away_club_goals"]],
        [0, 1],
        default=2,
    )
    data["goal_diff"] = data["home_club_goals"] - data["away_club_goals"]
    data["total_goals"] = data["home_club_goals"] + data["away_club_goals"]
    data["home_win"] = (data["result"] == 0).astype(int)
    data["away_win"] = (data["result"] == 2).astype(int)
    data["draw"] = (data["result"] == 1).astype(int)

    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data.attrs["ratings"] = ratings
    data.attrs["form"] = form
    return data


def train_test_split_by_date(matches: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = matches.sort_values("date").reset_index(drop=True)
    split_index = max(1, int(len(ordered) * (1.0 - test_fraction)))
    train = ordered.iloc[:split_index].copy()
    test = ordered.iloc[split_index:].copy()
    return train, test


def build_feature_lists(matches: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    static_diff_features = [
        column
        for column in matches.columns
        if column.endswith("_diff")
        and column not in {"goal_diff", "total_goals"}
    ]
    weather_features = [
        "weather_available",
        "weather_avg_temp_c",
        "weather_max_temp_c",
        "weather_indoor_ac",
        "weather_adjusted_temp_c",
    ]
    rolling_features = [
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_form_points_5",
        "away_form_points_5",
        "home_form_gd_5",
        "away_form_gd_5",
        "home_recent_goals_for_5",
        "away_recent_goals_for_5",
        "home_recent_goals_against_5",
        "away_recent_goals_against_5",
        "home_advantage",
    ]
    poisson_features = rolling_features + weather_features + [
        column for column in static_diff_features if column not in rolling_features and column not in weather_features
    ]
    classifier_features = rolling_features + weather_features + [
        column for column in static_diff_features if column not in rolling_features and column not in weather_features
    ]
    xgb_features = classifier_features.copy()
    return poisson_features, classifier_features, xgb_features


def make_numeric_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


from sklearn.model_selection import KFold


def train_models(
    matches: pd.DataFrame, feature_columns: Sequence[str]
) -> Dict[str, object]:
    # 1. Clean Time-Series Train/Test Split
    train, test = train_test_split_by_date(matches)

    # Prepare historical data slices
    X_train_base = train.loc[:, feature_columns].copy()
    X_test_base = test.loc[:, feature_columns].copy()

    y_home_goals_train = train["home_club_goals"]
    y_away_goals_train = train["away_club_goals"]
    y_result_train = train["result"]
    y_result_test = test["result"]

    # 2. GENERATE OUT-OF-FOLD (OOF) POISSON PREDICTIONS TO PREVENT LEAKAGE
    # This teaches XGBoost how to interpret the Poisson model's *imperfections*
    oof_home_lambdas = np.zeros(len(train))
    oof_away_lambdas = np.zeros(len(train))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X_train_base):
        # Fit internal Poisson fold pipelines
        fold_poisson_home = make_numeric_pipeline(
            PoissonRegressor(alpha=0.001, max_iter=5000)
        )
        fold_poisson_away = make_numeric_pipeline(
            PoissonRegressor(alpha=0.001, max_iter=5000)
        )

        fold_poisson_home.fit(
            X_train_base.iloc[train_idx], y_home_goals_train.iloc[train_idx]
        )
        fold_poisson_away.fit(
            X_train_base.iloc[train_idx], y_away_goals_train.iloc[train_idx]
        )

        # Populate Out-of-Fold prediction validations
        oof_home_lambdas[val_idx] = fold_poisson_home.predict(
            X_train_base.iloc[val_idx]
        )
        oof_away_lambdas[val_idx] = fold_poisson_away.predict(
            X_train_base.iloc[val_idx]
        )

    # 3. FIT FINAL PRODUCTION POISSON PIPELINES ON ALL TRAINING DATA
    poisson_home = make_numeric_pipeline(
        PoissonRegressor(alpha=0.001, max_iter=5000)
    )
    poisson_away = make_numeric_pipeline(
        PoissonRegressor(alpha=0.001, max_iter=5000)
    )
    poisson_home.fit(X_train_base, y_home_goals_train)
    poisson_away.fit(X_train_base, y_away_goals_train)

    # 4. ENRICH XGBOOST TRAINING AND TESTING MATRICES WITH POISSON OUTPUTS
    X_train_xgb = X_train_base.copy()
    X_train_xgb["poisson_home_lambda"] = oof_home_lambdas
    X_train_xgb["poisson_away_lambda"] = oof_away_lambdas

    # Generate test-set predictions using the final production Poisson instances
    X_test_xgb = X_test_base.copy()
    X_test_xgb["poisson_home_lambda"] = poisson_home.predict(X_test_base)
    X_test_xgb["poisson_away_lambda"] = poisson_away.predict(X_test_base)

    # 5. TRAIN LOGISTIC REGRESSION BASELINE
    logistic = make_numeric_pipeline(
        LogisticRegression(max_iter=4000, solver="lbfgs", C=1.0)
    )
    logistic.fit(X_train_base, y_result_train)

    # 6. TRAIN THE BLENDED XGBOOST PIPELINE
    xgb = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.85,
                    reg_lambda=1.5,
                    min_child_weight=2.0,
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=42,
                ),
            ),
        ]
    )
    xgb.fit(X_train_xgb, y_result_train)

    # 7. METRIC EVALUATIONS
    loss, accuracy = evaluate_poisson_classifier(
        poisson_home, poisson_away, X_test_base, y_result_test
    )

    train_metrics = {
        "poisson_log_loss": loss,
        "poisson_accuracy": accuracy,
        "logistic_log_loss": float(
            log_loss(
                y_result_test,
                logistic.predict_proba(X_test_base),
                labels=[0, 1, 2],
            )
        ),
        "logistic_accuracy": float(
            accuracy_score(y_result_test, logistic.predict(X_test_base))
        ),
        "xgb_log_loss": float(
            log_loss(
                y_result_test, xgb.predict_proba(X_test_xgb), labels=[0, 1, 2]
            )
        ),
        "xgb_accuracy": float(
            accuracy_score(y_result_test, xgb.predict(X_test_xgb))
        ),
    }

    return {
        "poisson_home": poisson_home,
        "poisson_away": poisson_away,
        "logistic": logistic,
        "xgb": xgb,
        "train_metrics": train_metrics,
        "train": train,
        "test": test,
    }


def poisson_result_probabilities(home_mean: float, away_mean: float, max_goals: int = 12) -> np.ndarray:
    home_grid = np.arange(0, max_goals + 1)
    away_grid = np.arange(0, max_goals + 1)
    home_probs = poisson.pmf(home_grid, home_mean)
    away_probs = poisson.pmf(away_grid, away_mean)
    joint = np.outer(home_probs, away_probs)
    joint = joint / joint.sum()
    p_home = float(np.tril(joint, k=-1).sum())
    p_draw = float(np.trace(joint))
    p_away = float(np.triu(joint, k=1).sum())
    total = p_home + p_draw + p_away
    return np.array([p_home / total, p_draw / total, p_away / total], dtype=float)


def evaluate_poisson_classifier(
    poisson_home, poisson_away, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[float, float]:
    """Evaluates the Poisson model and returns both Log Loss and Accuracy."""
    # 1. Predict the lambda (mean goals) for home and away teams
    home_means = poisson_home.predict(X_test)
    away_means = poisson_away.predict(X_test)

    # 2. Generate the match outcome probabilities matrix
    # Shape of proba will be (n_samples, 3) representing [Home Win, Draw, Away Win]
    proba = np.vstack(
        [
            poisson_result_probabilities(h, a)
            for h, a in zip(home_means, away_means)
        ]
    )

    # 3. Compute Log Loss (Requires probabilities)
    loss = float(log_loss(y_test, proba, labels=[0, 1, 2]))

    # 4. Compute Accuracy (Requires hard class predictions)
    # np.argmax gets the index (0, 1, or 2) of the highest probability for each row
    predictions = np.argmax(proba, axis=1)

    # Compare predictions to actual y_test labels
    accuracy = float(accuracy_score(y_test, predictions))

    # Returning both metrics as a tuple
    return loss, accuracy


# --- Example Usage ---
# loss, acc = evaluate_poisson_classifier(poisson_home, poisson_away, X_test, y_test)
# print(f"Log Loss: {loss:.4f} | Accuracy: {acc:.2%}")


def predict_matchup(
    models: Dict[str, object],
    match_row: pd.Series,
    method: str = "xgb",
) -> Dict[str, object]:
    feature_columns = list(models["feature_columns"])
    features_base = match_row.loc[feature_columns].to_frame().T

    if method == "poisson":
        home_mean = float(models["poisson_home"].predict(features_base)[0])
        away_mean = float(models["poisson_away"].predict(features_base)[0])
        proba = poisson_result_probabilities(home_mean, away_mean)
    elif method == "logistic":
        proba = models["logistic"].predict_proba(features_base)[0]
        home_mean = away_mean = np.nan
    elif method == "xgb":
        # Create stacked features dynamically for inference matching training shapes
        features_xgb = features_base.copy()
        home_mean = float(models["poisson_home"].predict(features_base)[0])
        away_mean = float(models["poisson_away"].predict(features_base)[0])

        features_xgb["poisson_home_lambda"] = home_mean
        features_xgb["poisson_away_lambda"] = away_mean

        proba = models["xgb"].predict_proba(features_xgb)[0]
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "home_team": match_row["home_team"],
        "away_team": match_row["away_team"],
        "prob_home_win": float(proba[0]),
        "prob_draw": float(proba[1]),
        "prob_away_win": float(proba[2]),
        "expected_home_goals": None if np.isnan(home_mean) else float(home_mean),
        "expected_away_goals": None if np.isnan(away_mean) else float(away_mean),
    }


def build_strength_table(team_features: pd.DataFrame) -> pd.DataFrame:
    table = team_features.copy()
    ranking_inverse = table["fifa_ranking_inverse"]
    table["combined_strength"] = (
        0.35 * table["fc26_top11_overall_mean"].fillna(table["fc26_top11_overall_mean"].median())
        + 0.20 * table["fc26_attack_index_mean"].fillna(table["fc26_attack_index_mean"].median())
        + 0.20 * table["fc26_defense_index_mean"].fillna(table["fc26_defense_index_mean"].median())
        + 0.15 * ranking_inverse.fillna(ranking_inverse.median())
        + 0.10 * table["fc26_gk_rating_max"].fillna(table["fc26_gk_rating_max"].median())
    )
    table = table.sort_values("combined_strength", ascending=False).reset_index(drop=True)
    return table


def simulate_knockout_bracket(
    teams: Sequence[str],
    matchup_predictor: Callable[[str, str], Dict[str, object]],
    n_simulations: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts: Counter[str] = Counter()

    if len(teams) % 2 != 0:
        raise ValueError("Knockout simulation requires an even number of teams.")

    for _ in range(n_simulations):
        current_round = list(teams)
        while len(current_round) > 1:
            next_round: List[str] = []
            for home, away in zip(current_round[0::2], current_round[1::2]):
                outcome = matchup_predictor(home, away)
                probs = np.array(
                    [outcome["prob_home_win"], outcome["prob_draw"], outcome["prob_away_win"]],
                    dtype=float,
                )
                probs = probs / probs.sum()
                sample = rng.choice([0, 1, 2], p=probs)
                if sample == 0:
                    winner = home
                elif sample == 2:
                    winner = away
                else:
                    winner = home if rng.random() < 0.5 else away
                next_round.append(winner)
            current_round = next_round
        counts[current_round[0]] += 1

    total = float(n_simulations)
    return pd.DataFrame(
        sorted(counts.items(), key=lambda item: item[1], reverse=True),
        columns=["team", "champion_probability"],
    ).assign(champion_probability=lambda df: df["champion_probability"] / total)


def prepare_models() -> Dict[str, object]:
    games, teams, players = load_sources()
    alias_map = build_alias_map(games, teams)
    team_meta = build_team_metadata(teams, alias_map)
    team_features = build_fc26_team_features(players, alias_map)
    team_features = team_features.merge(team_meta, on="team", how="left")
    population_features = build_population_features(alias_map)
    if not population_features.empty:
        team_features = team_features.merge(population_features, on="team", how="left")

    for column in TEAM_META_COLUMNS + ["total_market_value_log1p", "fifa_ranking_inverse"]:
        if column in team_features.columns:
            team_features[column] = pd.to_numeric(team_features[column], errors="coerce")

    for column in [
        "country_population",
        "country_population_millions",
        "country_population_log1p",
        "country_population_share_world",
    ]:
        if column in team_features.columns:
            team_features[column] = pd.to_numeric(team_features[column], errors="coerce")

    team_features["fifa_ranking"] = team_features["fifa_ranking"].fillna(team_features["fifa_ranking"].median())
    team_features["average_age"] = team_features["average_age"].fillna(team_features["average_age"].median())
    team_features["foreigners_percentage"] = team_features["foreigners_percentage"].fillna(
        team_features["foreigners_percentage"].median()
    )
    team_features["total_market_value_log1p"] = team_features["total_market_value_log1p"].fillna(
        team_features["total_market_value_log1p"].median()
    )
    team_features["fifa_ranking_inverse"] = team_features["fifa_ranking_inverse"].fillna(
        team_features["fifa_ranking_inverse"].median()
    )
    if "country_population" in team_features.columns:
        team_features["country_population"] = team_features["country_population"].fillna(
            team_features["country_population"].median()
        )
    if "country_population_millions" in team_features.columns:
        team_features["country_population_millions"] = team_features["country_population_millions"].fillna(
            team_features["country_population_millions"].median()
        )
    if "country_population_log1p" in team_features.columns:
        team_features["country_population_log1p"] = team_features["country_population_log1p"].fillna(
            team_features["country_population_log1p"].median()
        )
    if "country_population_share_world" in team_features.columns:
        team_features["country_population_share_world"] = team_features["country_population_share_world"].fillna(
            team_features["country_population_share_world"].median()
        )

    matches = build_match_dataset(games, team_features, alias_map)
    poisson_features, classifier_features, xgb_features = build_feature_lists(matches)
    current_state = summarize_current_state(matches.attrs["ratings"], matches.attrs["form"])
    current_state = current_state.merge(team_features, on="team", how="left")
    for column in current_state.select_dtypes(include=[np.number]).columns:
        current_state[column] = pd.to_numeric(current_state[column], errors="coerce")

    models = train_models(matches, classifier_features)
    models["feature_columns"] = classifier_features
    models["poisson_features"] = poisson_features
    models["xgb_features"] = xgb_features
    models["team_features"] = team_features
    models["current_state"] = current_state
    models["matches"] = matches
    models["alias_map"] = alias_map
    models["strength_table"] = build_strength_table(team_features)
    _, future_weather_context = build_weather_context()
    models["weather_context"] = future_weather_context
    return models


def demo_bracket_teams(strength_table: pd.DataFrame, size: int = 16) -> List[str]:
    return strength_table.head(size)["team"].tolist()


def run_demo() -> None:
    models = prepare_models()
    metrics = models["train_metrics"]
    strength_table = models["strength_table"]
    feature_columns = models["feature_columns"]
    matches = models["matches"]

    print("Model metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nTop teams by combined strength")
    print(strength_table[["team", "combined_strength", "fifa_ranking", "fc26_top11_overall_mean"]].head(15).to_string(index=False))

    bracket = demo_bracket_teams(strength_table, size=16)

    def make_matchup_predictor(method: str) -> Callable[[str, str], Dict[str, object]]:
        def predict(home: str, away: str) -> Dict[str, object]:
            row = build_match_context(models, home, away)
            return predict_matchup(models, row, method=method)

        return predict

    sims = list()
    print("\nDemo knockout champion probabilities")
    for method in ["poisson", "logistic", "xgb"]:
        sim = simulate_knockout_bracket(bracket, make_matchup_predictor(method), n_simulations=3000)
        sims.append(sim)
        print(f"\n{method.upper()}")
        print(sim.head(10).to_string(index=False))

    all_simulations = pd.concat(sims)


def build_match_context(
    models: Dict[str, object], home: str, away: str, venue_city: str = None
) -> pd.Series:
    team_features = models["team_features"].set_index("team")
    state_features = models["current_state"].set_index("team")

    sample = {name: 0.0 for name in models["feature_columns"]}
    sample["home_team"] = home
    sample["away_team"] = away
    sample["home_advantage"] = (
        1.0 if home in {"Canada", "Mexico", "United States"} else 0.0
    )

    home_row = (
        team_features.reindex([home]).iloc[0]
        if home in team_features.index
        else pd.Series(dtype=float)
    )
    away_row = (
        team_features.reindex([away]).iloc[0]
        if away in team_features.index
        else pd.Series(dtype=float)
    )
    home_state = (
        state_features.reindex([home]).iloc[0]
        if home in state_features.index
        else pd.Series(dtype=float)
    )
    away_state = (
        state_features.reindex([away]).iloc[0]
        if away in state_features.index
        else pd.Series(dtype=float)
    )

    for col in team_features.columns:
        if col == "team":
            continue
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        diff_col = f"{col}_diff"
        if home_col in sample:
            sample[home_col] = float(home_row.get(col, np.nan))
        if away_col in sample:
            sample[away_col] = float(away_row.get(col, np.nan))
        if diff_col in sample:
            sample[diff_col] = float(home_row.get(col, np.nan)) - float(
                away_row.get(col, np.nan)
            )

    sample["home_elo"] = float(home_state.get("current_elo", np.nan))
    sample["away_elo"] = float(away_state.get("current_elo", np.nan))
    sample["elo_diff"] = sample["home_elo"] - sample["away_elo"]
    sample["home_form_points_5"] = float(
        home_state.get("current_form_points_5", np.nan)
    )
    sample["away_form_points_5"] = float(
        away_state.get("current_form_points_5", np.nan)
    )
    sample["home_form_gd_5"] = float(
        home_state.get("current_form_gd_5", np.nan)
    )
    sample["away_form_gd_5"] = float(
        away_state.get("current_form_gd_5", np.nan)
    )
    sample["home_recent_goals_for_5"] = float(
        home_state.get("current_recent_goals_for_5", np.nan)
    )
    sample["away_recent_goals_for_5"] = float(
        away_state.get("current_recent_goals_for_5", np.nan)
    )
    sample["home_recent_goals_against_5"] = float(
        home_state.get("current_recent_goals_against_5", np.nan)
    )
    sample["away_recent_goals_against_5"] = float(
        away_state.get("current_recent_goals_against_5", np.nan)
    )

    # ALIGN WEATHER CONTEXT DYNAMICALLY TO SPECIFIC HOST CITIES IF SUPPLIED
    future_path = resolve_optional_data_path(WEATHER_2026_FILENAME)
    if future_path and venue_city:
        future_df = pd.read_csv(future_path)
        city_match = future_df[
            future_df["City"].str.lower() == venue_city.lower()
        ]
        if not city_match.empty:
            forecast = pd.to_numeric(
                city_match.iloc[0]["2026 Summer Forecast Estimate (°C)"],
                errors="coerce",
            )
            roof = parse_indoor_flag(
                city_match.iloc[0]["Indoor / Retractable Roof?"]
            )

            sample["weather_available"] = 1.0
            sample["weather_avg_temp_c"] = forecast
            sample["weather_max_temp_c"] = forecast
            sample["weather_indoor_ac"] = roof
            sample["weather_adjusted_temp_c"] = (
                forecast - (4.0 * roof) if roof == 1.0 else forecast
            )
            return pd.Series(sample)

    # Default fallback to the global tournament-wide means if city lacks data
    weather_context = models.get("weather_context", {})
    sample["weather_available"] = float(
        weather_context.get("weather_available", np.nan)
    )
    sample["weather_avg_temp_c"] = float(
        weather_context.get("weather_avg_temp_c", np.nan)
    )
    sample["weather_max_temp_c"] = float(
        weather_context.get("weather_max_temp_c", np.nan)
    )
    sample["weather_indoor_ac"] = float(
        weather_context.get("weather_indoor_ac", np.nan)
    )
    sample["weather_adjusted_temp_c"] = float(
        weather_context.get("weather_adjusted_temp_c", np.nan)
    )
    return pd.Series(sample)


if __name__ == "__main__":
    run_demo()
