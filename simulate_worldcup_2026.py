from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from worldcup_models import (
    build_match_context,
    canonicalize_team,
    predict_matchup,
    prepare_models,
)


QUALIFIED_TEAMS = [
    "Canada",
    "Mexico",
    "USA",
    "Australia",
    "Iraq",
    "IR Iran",
    "Japan",
    "Jordan",
    "Korea Republic",
    "Qatar",
    "Saudi Arabia",
    "Uzbekistan",
    "Algeria",
    "Cabo Verde",
    "Congo DR",
    "Côte d'Ivoire",
    "Egypt",
    "Ghana",
    "Morocco",
    "Senegal",
    "South Africa",
    "Tunisia",
    "Curaçao",
    "Haiti",
    "Panama",
    "Argentina",
    "Brazil",
    "Colombia",
    "Ecuador",
    "Paraguay",
    "Uruguay",
    "New Zealand",
    "Austria",
    "Belgium",
    "Bosnia and Herzegovina",
    "Croatia",
    "Czechia",
    "England",
    "France",
    "Germany",
    "Netherlands",
    "Norway",
    "Portugal",
    "Scotland",
    "Spain",
    "Sweden",
    "Switzerland",
    "Türkiye",
]

HOSTS = ["Canada", "Mexico", "United States"]
GROUP_NAMES = list("ABCDEFGHIJKL")
METHODS = ("poisson", "logistic", "xgb")
OUTPUT_DIR = Path("worldcup_2026_outputs")
DATA_DIR = Path("data")
DEFAULT_GROUP_DRAW_PATH = DATA_DIR / "group_draw_2026.csv"

QUALIFIED_TEAM_CONFEDS = {
    "Canada": "CONCACAF",
    "Mexico": "CONCACAF",
    "United States": "CONCACAF",
    "Australia": "AFC",
    "Iraq": "AFC",
    "Iran": "AFC",
    "Japan": "AFC",
    "Jordan": "AFC",
    "South Korea": "AFC",
    "Qatar": "AFC",
    "Saudi Arabia": "AFC",
    "Uzbekistan": "AFC",
    "Algeria": "CAF",
    "Cape Verde": "CAF",
    "Democratic Republic of the Congo": "CAF",
    "Ivory Coast": "CAF",
    "Egypt": "CAF",
    "Ghana": "CAF",
    "Morocco": "CAF",
    "Senegal": "CAF",
    "South Africa": "CAF",
    "Tunisia": "CAF",
    "Curaçao": "CONCACAF",
    "Haiti": "CONCACAF",
    "Panama": "CONCACAF",
    "Argentina": "CONMEBOL",
    "Brazil": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    "New Zealand": "OFC",
    "Austria": "UEFA",
    "Belgium": "UEFA",
    "Bosnia-Herzegovina": "UEFA",
    "Croatia": "UEFA",
    "Czech Republic": "UEFA",
    "England": "UEFA",
    "France": "UEFA",
    "Germany": "UEFA",
    "Netherlands": "UEFA",
    "Norway": "UEFA",
    "Portugal": "UEFA",
    "Scotland": "UEFA",
    "Spain": "UEFA",
    "Sweden": "UEFA",
    "Switzerland": "UEFA",
    "Turkiye": "UEFA",
}


def canonical_team_list(models: Dict[str, object], team_names: Iterable[str]) -> List[str]:
    alias_map = models["alias_map"]
    canonical = [canonicalize_team(team, alias_map) for team in team_names]
    canonical = [team for team in canonical if team]
    return canonical


def build_qualified_table(models: Dict[str, object], team_names: Iterable[str]) -> pd.DataFrame:
    canonical = canonical_team_list(models, team_names)
    strength = models["strength_table"].copy()
    table = strength[strength["team"].isin(canonical)].copy()
    table = table.drop_duplicates(subset=["team"]).copy()
    missing = [team for team in canonical if team not in set(table["team"])]
    if missing:
        raise ValueError(f"Missing teams in model data: {missing}")
    table["qualified_input_order"] = table["team"].map({team: i for i, team in enumerate(canonical)})
    table["confederation"] = table["team"].map(QUALIFIED_TEAM_CONFEDS).fillna(table["confederation"])
    return table.sort_values("qualified_input_order").reset_index(drop=True)


def build_pots(table: pd.DataFrame) -> List[List[str]]:
    ranked = table.sort_values("combined_strength", ascending=False)["team"].tolist()
    hosts = [team for team in HOSTS if team in ranked]
    rest = [team for team in ranked if team not in hosts]
    pot1 = hosts + rest[: 12 - len(hosts)]
    remaining = [team for team in ranked if team not in pot1]
    pots = [pot1]
    for start in range(0, len(remaining), 12):
        pots.append(remaining[start : start + 12])
    if len(pots) != 4 or any(len(pot) != 12 for pot in pots):
        raise ValueError("Could not build 4 pots of 12 teams.")
    return pots


def team_confederation_lookup(table: pd.DataFrame) -> Dict[str, str]:
    return table.set_index("team")["confederation"].to_dict()


def draw_groups(table: pd.DataFrame, seed: int = 42) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    pots = build_pots(table)
    confed = team_confederation_lookup(table)

    for _attempt in range(500):
        groups: Dict[str, List[str]] = {group: [] for group in GROUP_NAMES}
        confed_counts = {group: Counter() for group in GROUP_NAMES}
        success = True

        for pot in pots:
            pot_order = pot.copy()
            rng.shuffle(pot_order)
            for team in pot_order:
                team_confed = confed.get(team, "UNKNOWN")
                valid_groups = []
                for group in GROUP_NAMES:
                    if len(groups[group]) >= 4:
                        continue
                    current = confed_counts[group][team_confed]
                    if team_confed == "UEFA":
                        if current < 2:
                            valid_groups.append(group)
                    else:
                        if current < 1:
                            valid_groups.append(group)

                if not valid_groups:
                    success = False
                    break

                min_confed = min(confed_counts[group][team_confed] for group in valid_groups)
                candidates = [group for group in valid_groups if confed_counts[group][team_confed] == min_confed]
                min_size = min(len(groups[group]) for group in candidates)
                candidates = [group for group in candidates if len(groups[group]) == min_size]
                chosen = rng.choice(candidates)
                groups[chosen].append(team)
                confed_counts[chosen][team_confed] += 1

            if not success:
                break

        if success and all(len(group) == 4 for group in groups.values()):
            return groups

    raise RuntimeError("Unable to draw groups with the current constraints after many attempts.")


def precompute_matchups(models: Dict[str, object], teams: Iterable[str]) -> Dict[str, Dict[Tuple[str, str], Dict[str, float]]]:
    team_list = list(teams)
    cache: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = {method: {} for method in METHODS}
    for home in team_list:
        for away in team_list:
            if home == away:
                continue
            row = build_match_context(models, home, away)
            for method in METHODS:
                pred = predict_matchup(models, row, method=method)
                cache[method][(home, away)] = pred
    return cache


def sample_conditioned_score(
    rng: np.random.Generator,
    home_mean: float,
    away_mean: float,
    outcome: int,
    max_attempts: int = 250,
) -> Tuple[int, int]:
    home_mean = max(0.05, float(home_mean))
    away_mean = max(0.05, float(away_mean))

    for _ in range(max_attempts):
        home_goals = int(rng.poisson(home_mean))
        away_goals = int(rng.poisson(away_mean))
        if outcome == 0 and home_goals > away_goals:
            return home_goals, away_goals
        if outcome == 1 and home_goals == away_goals:
            return home_goals, away_goals
        if outcome == 2 and home_goals < away_goals:
            return home_goals, away_goals

    base_home = max(0, int(round(home_mean)))
    base_away = max(0, int(round(away_mean)))
    if outcome == 0:
        if base_home <= base_away:
            base_home = base_away + 1
        return base_home, base_away
    if outcome == 2:
        if base_away <= base_home:
            base_away = base_home + 1
        return base_home, base_away
    base = max(0, int(round((home_mean + away_mean) / 2.0)))
    return base, base


def play_match(
    rng: np.random.Generator,
    method: str,
    home: str,
    away: str,
    matchups: Dict[str, Dict[Tuple[str, str], Dict[str, float]]],
    knockout: bool = False,
) -> Tuple[int, int, str]:
    poisson_pred = matchups["poisson"][(home, away)]
    home_mean = float(poisson_pred["expected_home_goals"])
    away_mean = float(poisson_pred["expected_away_goals"])

    if method == "poisson":
        home_goals = int(rng.poisson(max(0.05, home_mean)))
        away_goals = int(rng.poisson(max(0.05, away_mean)))
    else:
        pred = matchups[method][(home, away)]
        probs = np.array(
            [pred["prob_home_win"], pred["prob_draw"], pred["prob_away_win"]],
            dtype=float,
        )
        probs = probs / probs.sum()
        outcome = int(rng.choice([0, 1, 2], p=probs))
        home_goals, away_goals = sample_conditioned_score(rng, home_mean, away_mean, outcome)

    if knockout and home_goals == away_goals:
        winner = home if rng.random() < 0.5 else away
    else:
        if home_goals > away_goals:
            winner = home
        elif home_goals < away_goals:
            winner = away
        else:
            winner = home if rng.random() < 0.5 else away

    return home_goals, away_goals, winner


def simulate_group_stage(
    rng: np.random.Generator,
    method: str,
    groups: Dict[str, List[str]],
    matchups: Dict[str, Dict[Tuple[str, str], Dict[str, float]]],
    strength_lookup: Dict[str, float],
) -> pd.DataFrame:
    records = []

    for group_name, teams in groups.items():
        standings = {team: {"points": 0, "gf": 0, "ga": 0} for team in teams}
        for home, away in combinations(teams, 2):
            home_goals, away_goals, _ = play_match(rng, method, home, away, matchups, knockout=False)
            standings[home]["gf"] += home_goals
            standings[home]["ga"] += away_goals
            standings[away]["gf"] += away_goals
            standings[away]["ga"] += home_goals

            if home_goals > away_goals:
                standings[home]["points"] += 3
            elif home_goals < away_goals:
                standings[away]["points"] += 3
            else:
                standings[home]["points"] += 1
                standings[away]["points"] += 1

        for team, stats in standings.items():
            records.append(
                {
                    "group": group_name,
                    "team": team,
                    "points": stats["points"],
                    "gd": stats["gf"] - stats["ga"],
                    "gf": stats["gf"],
                    "strength": strength_lookup.get(team, 0.0),
                }
            )

    table = pd.DataFrame(records)
    table = table.sort_values(
        ["group", "points", "gd", "gf", "strength", "team"],
        ascending=[True, False, False, False, False, True],
    ).reset_index(drop=True)
    table["group_rank"] = table.groupby("group").cumcount() + 1
    return table


def assign_third_places(qualified_groups: List[str]) -> Dict[str, str]:
    ALLOWED_3RDS = {
        'E': ['A', 'B', 'C', 'D', 'F'],
        'I': ['C', 'D', 'F', 'G', 'H'],
        'A': ['C', 'E', 'F', 'H', 'I'],
        'L': ['E', 'H', 'I', 'J', 'K'],
        'G': ['A', 'E', 'H', 'I', 'J'],
        'D': ['B', 'E', 'F', 'I', 'J'],
        'B': ['E', 'F', 'G', 'I', 'J'],
        'K': ['D', 'E', 'I', 'J', 'L']
    }
    winners = ['E', 'I', 'A', 'L', 'G', 'D', 'B', 'K']
    matching = {}
    used_3rds = set()
    sorted_groups = sorted(qualified_groups)
    
    def backtrack(winner_idx):
        if winner_idx == len(winners):
            return True
        w = winners[winner_idx]
        for t in ALLOWED_3RDS[w]:
            if t in sorted_groups and t not in used_3rds and t != w:
                matching[w] = t
                used_3rds.add(t)
                if backtrack(winner_idx + 1):
                    return True
                used_3rds.remove(t)
                del matching[w]
        return False
        
    if backtrack(0):
        return matching
        
    used_3rds.clear()
    matching.clear()
    def backtrack_relaxed(winner_idx):
        if winner_idx == len(winners):
            return True
        w = winners[winner_idx]
        for t in ALLOWED_3RDS[w]:
            if t in sorted_groups and t not in used_3rds:
                matching[w] = t
                used_3rds.add(t)
                if backtrack_relaxed(winner_idx + 1):
                    return True
                used_3rds.remove(t)
                del matching[w]
        return False
        
    if backtrack_relaxed(0):
        return matching
        
    return {w: sorted_groups[i % len(sorted_groups)] for i, w in enumerate(winners)}


def select_knockout_matchups(group_table: pd.DataFrame) -> List[Tuple[str, str]]:
    winners = {}
    runners_up = {}
    thirds = []
    
    for group, group_df in group_table.groupby("group"):
        group_df = group_df.sort_values("group_rank")
        winners[group] = group_df.iloc[0]["team"]
        runners_up[group] = group_df.iloc[1]["team"]
        thirds.append({
            "group": group,
            "team": group_df.iloc[2]["team"],
            "points": group_df.iloc[2]["points"],
            "gd": group_df.iloc[2]["gd"],
            "gf": group_df.iloc[2]["gf"],
            "strength": group_df.iloc[2]["strength"]
        })
        
    thirds_df = pd.DataFrame(thirds)
    thirds_df = thirds_df.sort_values(
        ["points", "gd", "gf", "strength", "group"],
        ascending=[False, False, False, False, True]
    )
    best_thirds = thirds_df.head(8)
    
    qualified_thirds_groups = best_thirds["group"].tolist()
    thirds_mapping = assign_third_places(qualified_thirds_groups)
    thirds_by_group = best_thirds.set_index("group")["team"].to_dict()
    
    matches = [
        (winners["E"], thirds_by_group[thirds_mapping["E"]]),
        (winners["I"], thirds_by_group[thirds_mapping["I"]]),
        (runners_up["A"], runners_up["B"]),
        (winners["F"], runners_up["C"]),
        (winners["A"], thirds_by_group[thirds_mapping["A"]]),
        (winners["L"], thirds_by_group[thirds_mapping["L"]]),
        (runners_up["K"], runners_up["L"]),
        (winners["H"], runners_up["J"]),
        (winners["C"], runners_up["F"]),
        (runners_up["E"], runners_up["I"]),
        (winners["D"], thirds_by_group[thirds_mapping["D"]]),
        (winners["G"], thirds_by_group[thirds_mapping["G"]]),
        (winners["B"], thirds_by_group[thirds_mapping["B"]]),
        (winners["K"], thirds_by_group[thirds_mapping["K"]]),
        (winners["J"], runners_up["H"]),
        (runners_up["D"], runners_up["G"])
    ]
    return matches


def select_knockout_teams(group_table: pd.DataFrame) -> pd.DataFrame:
    top_two = group_table[group_table["group_rank"] <= 2].copy()
    thirds = group_table[group_table["group_rank"] == 3].copy()
    thirds = thirds.sort_values(["points", "gd", "gf", "strength", "team"], ascending=[False, False, False, False, True]).head(8)
    qualifiers = pd.concat([top_two, thirds], ignore_index=True)
    qualifiers = qualifiers.sort_values(["points", "gd", "gf", "strength", "team"], ascending=[False, False, False, False, True]).reset_index(drop=True)
    qualifiers["seed"] = np.arange(1, len(qualifiers) + 1)
    return qualifiers


def simulate_knockout(
    rng: np.random.Generator,
    method: str,
    group_table: pd.DataFrame,
    matchups: Dict[str, Dict[Tuple[str, str], Dict[str, float]]],
) -> str:
    r32_pairings = select_knockout_matchups(group_table)
    current_round = []
    for home, away in r32_pairings:
        current_round.extend([home, away])

    while len(current_round) > 1:
        next_round = []
        for home, away in zip(current_round[0::2], current_round[1::2]):
            _, _, winner = play_match(rng, method, home, away, matchups, knockout=True)
            next_round.append(winner)
        current_round = next_round

    return current_round[0]


def run_monte_carlo(
    models: Dict[str, object],
    groups: Dict[str, List[str]],
    method: str,
    n_sims: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    teams = sorted({team for group in groups.values() for team in group})
    matchups = precompute_matchups(models, teams)
    strength_lookup = models["strength_table"].set_index("team")["combined_strength"].to_dict()

    champion_counts: Counter[str] = Counter()
    finalist_counts: Counter[str] = Counter()
    semifinal_counts: Counter[str] = Counter()

    for _ in range(n_sims):
        group_table = simulate_group_stage(rng, method, groups, matchups, strength_lookup)
        r32_pairings = select_knockout_matchups(group_table)
        current_round = []
        for home, away in r32_pairings:
            current_round.extend([home, away])

        while len(current_round) > 1:
            next_round = []
            for home, away in zip(current_round[0::2], current_round[1::2]):
                _, _, winner = play_match(rng, method, home, away, matchups, knockout=True)
                next_round.append(winner)
            if len(current_round) == 4:
                semifinal_counts.update(current_round)
            if len(current_round) == 2:
                finalist_counts.update(current_round)
            current_round = next_round

        champion_counts.update(current_round)

    all_teams = sorted(teams)
    champions = (
        pd.DataFrame(champion_counts.items(), columns=["team", "champion_prob"])
        .assign(champion_prob=lambda df: df["champion_prob"] / float(n_sims))
        .sort_values("champion_prob", ascending=False)
        .reset_index(drop=True)
    )
    stage_probs = pd.DataFrame(
        {
            "team": all_teams,
            "semi_final_prob": [semifinal_counts[t] / float(n_sims) for t in all_teams],
            "final_prob": [finalist_counts[t] / float(n_sims) for t in all_teams],
            "champion_prob": [champion_counts[t] / float(n_sims) for t in all_teams],
        }
    ).sort_values("champion_prob", ascending=False)
    return champions, stage_probs


def format_group_draw(groups: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for group, teams in groups.items():
        for position, team in enumerate(teams, start=1):
            rows.append({"group": group, "position": position, "team": team})
    return pd.DataFrame(rows).sort_values(["group", "position"]).reset_index(drop=True)


def load_group_draw_from_csv(path: Path, alias_map: Dict[str, str] | None = None) -> Dict[str, List[str]]:
    table = pd.read_csv(path)
    required = {"group", "position", "team"}
    if not required.issubset(set(table.columns)):
        missing = required - set(table.columns)
        raise ValueError(f"Group draw file missing columns: {sorted(missing)}")
    groups: Dict[str, List[str]] = {}
    for group, sub in table.sort_values(["group", "position"]).groupby("group"):
        teams = sub["team"].tolist()
        if alias_map is not None:
            teams = [canonicalize_team(team, alias_map) for team in teams]
        groups[group] = [team for team in teams if team]
    if len(groups) != 12 or any(len(teams) != 4 for teams in groups.values()):
        raise ValueError("Group draw CSV must contain 12 groups with 4 teams each.")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the FIFA World Cup 2026.")
    parser.add_argument("--sims", type=int, default=10000, help="Monte Carlo simulations per model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS), help="Model methods to run.")
    parser.add_argument("--group-draw", type=Path, default=DEFAULT_GROUP_DRAW_PATH, help="CSV file with group assignment.")
    args = parser.parse_args()

    models = prepare_models()
    qualified_table = build_qualified_table(models, QUALIFIED_TEAMS)
    if args.group_draw.exists():
        groups = load_group_draw_from_csv(args.group_draw, models["alias_map"])
    else:
        groups = draw_groups(qualified_table, seed=args.seed)
    draw_table = format_group_draw(groups)

    OUTPUT_DIR.mkdir(exist_ok=True)
    draw_table.to_csv(OUTPUT_DIR / "group_draw.csv", index=False)

    print("GROUP DRAW")
    for group, teams in groups.items():
        print(f"{group}: {', '.join(teams)}")

    print("\nMODEL RANKING OF QUALIFIED TEAMS")
    print(
        qualified_table[["team", "confederation", "combined_strength", "fifa_ranking"]]
        .sort_values("combined_strength", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    for method in args.methods:
        champions, stage_probs = run_monte_carlo(models, groups, method, args.sims, args.seed + 100)
        champions.to_csv(OUTPUT_DIR / f"champions_{method}.csv", index=False)
        stage_probs.to_csv(OUTPUT_DIR / f"stage_probs_{method}.csv", index=False)

        print(f"\n{method.upper()} - TOP CHAMPION PROBABILITIES")
        print(champions.head(10).to_string(index=False, formatters={"champion_prob": "{:.1%}".format}))

        print(f"\n{method.upper()} - TOP FINALISTS / SEMIFINALISTS")
        print(
            stage_probs.head(10).to_string(
                index=False,
                formatters={
                    "semi_final_prob": "{:.1%}".format,
                    "final_prob": "{:.1%}".format,
                    "champion_prob": "{:.1%}".format,
                },
            )
        )


if __name__ == "__main__":
    main()
