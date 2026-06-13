from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from simulate_worldcup_2026 import (
    QUALIFIED_TEAMS,
    build_qualified_table,
    draw_groups,
    load_group_draw_from_csv,
    precompute_matchups,
    select_knockout_teams,
    select_knockout_matchups,
)
from worldcup_models import prepare_models


OUTPUT_DIR = Path("worldcup_2026_outputs")
GROUP_DRAW_PATH = Path("data") / "group_draw_2026.csv"
METHODS = ("poisson", "logistic", "xgb")


@dataclass
class MatchNode:
    round_name: str
    home: str
    away: str
    home_goals: int
    away_goals: int
    winner: str
    x: float
    y: float
    child_left: "MatchNode | None" = None
    child_right: "MatchNode | None" = None


def score_line(node: MatchNode) -> str:
    if node.home_goals == node.away_goals:
        return f"{node.home_goals}-{node.away_goals} (pens)"
    return f"{node.home_goals}-{node.away_goals}"


def outcome_score(pred: Dict[str, object], outcome: int) -> Tuple[int, int]:
    home_mean = pred["expected_home_goals"] if pred["expected_home_goals"] is not None else 1.4
    away_mean = pred["expected_away_goals"] if pred["expected_away_goals"] is not None else 1.2
    if outcome == 0:
        home_goals = max(1, int(round(home_mean)))
        away_goals = max(0, home_goals - 1)
        if home_goals <= away_goals:
            home_goals = away_goals + 1
        return home_goals, away_goals
    if outcome == 2:
        away_goals = max(1, int(round(away_mean)))
        home_goals = max(0, away_goals - 1)
        if away_goals <= home_goals:
            away_goals = home_goals + 1
        return home_goals, away_goals
    base = max(0, int(round((home_mean + away_mean) / 2)))
    return base, base


def likely_outcome(pred: Dict[str, object]) -> int:
    probs = np.array(
        [pred["prob_home_win"], pred["prob_draw"], pred["prob_away_win"]],
        dtype=float,
    )
    return int(np.argmax(probs))


def likely_winner(pred: Dict[str, object], home: str, away: str, outcome: int, strength_lookup: Dict[str, float]) -> str:
    if outcome == 0:
        return home
    if outcome == 2:
        return away
    home_adv = float(pred["prob_home_win"])
    away_adv = float(pred["prob_away_win"])
    if home_adv > away_adv:
        return home
    if away_adv > home_adv:
        return away
    return home if strength_lookup.get(home, 0.0) >= strength_lookup.get(away, 0.0) else away


def build_likely_run(method: str) -> Tuple[Dict[str, List[str]], List[List[MatchNode]], MatchNode, pd.DataFrame, pd.DataFrame]:
    models = prepare_models()
    qualified = build_qualified_table(models, QUALIFIED_TEAMS)
    if GROUP_DRAW_PATH.exists():
        groups = load_group_draw_from_csv(GROUP_DRAW_PATH, models["alias_map"])
    else:
        groups = draw_groups(qualified, seed=202)
    teams = sorted({team for group in groups.values() for team in group})
    matchups = precompute_matchups(models, teams)
    strength_lookup = models["strength_table"].set_index("team")["combined_strength"].to_dict()

    group_match_rows: List[Dict[str, object]] = []
    group_standings_rows: List[Dict[str, object]] = []
    group_table_rows: List[Dict[str, object]] = []

    for group_name, group_teams in groups.items():
        standings = {team: {"points": 0, "gf": 0, "ga": 0} for team in group_teams}
        for home, away in combinations(group_teams, 2):
            pred = matchups[method][(home, away)]
            outcome = likely_outcome(pred)
            home_goals, away_goals = outcome_score(pred, outcome)
            winner = likely_winner(pred, home, away, outcome, strength_lookup)
            group_match_rows.append(
                {
                    "stage": "Group Stage",
                    "round": group_name,
                    "home": home,
                    "away": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "winner": winner,
                }
            )
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

        ranked = sorted(
            standings.items(),
            key=lambda item: (
                item[1]["points"],
                item[1]["gf"] - item[1]["ga"],
                item[1]["gf"],
                strength_lookup.get(item[0], 0.0),
                item[0],
            ),
            reverse=True,
        )
        for position, (team, stats) in enumerate(ranked, start=1):
            row = {
                "group": group_name,
                "team": team,
                "points": stats["points"],
                "gd": stats["gf"] - stats["ga"],
                "gf": stats["gf"],
                "strength": strength_lookup.get(team, 0.0),
                "group_rank": position,
            }
            group_table_rows.append(row)
            group_standings_rows.append(row | {"qualified": position <= 2})

    group_table = pd.DataFrame(group_table_rows)
    r32_pairings = select_knockout_matchups(group_table)
    current = []
    for home, away in r32_pairings:
        current.extend([home, away])

    round_names = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
    nodes_by_round: List[List[MatchNode]] = []
    round_index = 0
    while len(current) > 1:
        round_nodes: List[MatchNode] = []
        next_round: List[str] = []
        for home, away in zip(current[0::2], current[1::2]):
            pred = matchups[method][(home, away)]
            outcome = likely_outcome(pred)
            home_goals, away_goals = outcome_score(pred, outcome)
            winner = likely_winner(pred, home, away, outcome, strength_lookup)
            node = MatchNode(
                round_name=round_names[round_index],
                home=home,
                away=away,
                home_goals=home_goals,
                away_goals=away_goals,
                winner=winner,
                x=float(round_index),
                y=float(len(round_nodes)),
            )
            round_nodes.append(node)
            next_round.append(winner)
        nodes_by_round.append(round_nodes)
        current = next_round
        round_index += 1

    for i in range(1, len(nodes_by_round)):
        prev = nodes_by_round[i - 1]
        curr = nodes_by_round[i]
        for j, node in enumerate(curr):
            node.child_left = prev[2 * j]
            node.child_right = prev[2 * j + 1]
            node.y = (node.child_left.y + node.child_right.y) / 2.0

    champion = nodes_by_round[-1][0]
    match_rows = list(group_match_rows)
    for round_nodes in nodes_by_round:
        for node in round_nodes:
            match_rows.append(
                {
                    "stage": node.round_name,
                    "round": node.round_name,
                    "home": node.home,
                    "away": node.away,
                    "home_goals": node.home_goals,
                    "away_goals": node.away_goals,
                    "winner": node.winner,
                }
            )

    return groups, nodes_by_round, champion, pd.DataFrame(match_rows), pd.DataFrame(group_standings_rows)


def draw_bracket(nodes_by_round: List[List[MatchNode]], champion: MatchNode, output_path: Path, title: str) -> None:
    round_names = [nodes[0].round_name for nodes in nodes_by_round]
    x_positions = list(range(len(nodes_by_round)))

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(-0.6, len(nodes_by_round) - 0.1)
    ax.set_ylim(-1, max(15, max(node.y for node in nodes_by_round[0]) + 1))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    def node_box(node: MatchNode, is_champion: bool = False):
        x = node.x
        y = node.y
        width = 0.78
        height = 0.72
        fill = "#f9fafb"
        edge = "#d1d5db"
        if is_champion:
            fill = "#fff7d6"
            edge = "#d4a017"
        elif node.winner == node.home:
            fill = "#eff6ff"
            edge = "#3b82f6"
        else:
            fill = "#f0fdf4"
            edge = "#22c55e"
        rect = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.4,
            facecolor=fill,
            edgecolor=edge,
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.16, f"{node.home} vs {node.away}", ha="center", va="center", fontsize=8.6, fontweight="bold")
        ax.text(x, y - 0.02, score_line(node), ha="center", va="center", fontsize=10, color="#111827")
        ax.text(x, y - 0.19, f"Winner: {node.winner}", ha="center", va="center", fontsize=8.4, color="#374151")

    for round_index, round_nodes in enumerate(nodes_by_round):
        x = x_positions[round_index]
        ax.text(x, max(15, max(node.y for node in nodes_by_round[0]) + 0.5), round_names[round_index], ha="center", va="bottom", fontsize=13, fontweight="bold")
        for node in round_nodes:
            node.x = x
            node_box(node, is_champion=(node is champion))
            if node.child_left and node.child_right:
                for child in [node.child_left, node.child_right]:
                    ax.add_line(Line2D([child.x + 0.4, node.x - 0.4], [child.y, node.y], color="#9ca3af", linewidth=1.0))
                    ax.add_line(Line2D([child.x + 0.4, child.x + 0.55], [child.y, child.y], color="#9ca3af", linewidth=1.0))
                ax.add_line(Line2D([node.x - 0.55, node.x - 0.4], [node.y, node.y], color="#9ca3af", linewidth=1.0))

    ax.text(len(nodes_by_round) - 1, champion.y - 0.9, f"Champion: {champion.winner}", ha="center", va="top", fontsize=16, fontweight="bold", color="#92400e")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=18)
    plt.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_rows = []
    for method in METHODS:
        groups, nodes_by_round, champion, matches_df, standings_df = build_likely_run(method)
        prefix = f"likely_{method}"
        matches_path = OUTPUT_DIR / f"{prefix}_table.csv"
        standings_path = OUTPUT_DIR / f"{prefix}_group_standings.csv"
        svg_path = OUTPUT_DIR / f"{prefix}.svg"
        group_svg_path = OUTPUT_DIR / f"{prefix}_group_standings.svg"
        matches_df.to_csv(matches_path, index=False)
        standings_df.to_csv(standings_path, index=False)
        draw_bracket(nodes_by_round, champion, svg_path, title=f"Most Likely Path - {method.upper()}")
        # Reuse the group standings visual style from the existing SVG logic.
        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
        axes = axes.flatten()
        for ax, group in zip(axes, sorted(standings_df["group"].unique())):
            ax.axis("off")
            sub = standings_df[standings_df["group"] == group].sort_values(["group_rank", "points", "gd", "gf"], ascending=[True, False, False, False])
            display_df = sub[["group_rank", "team", "points", "gd", "gf", "qualified"]].copy()
            display_df.columns = ["Rank", "Team", "Pts", "GD", "GF", "Q"]
            table = ax.table(cellText=display_df.values, colLabels=display_df.columns, cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.45)
            ax.set_title(f"Group {group}", fontsize=13, fontweight="bold", pad=12)
            for row_idx in range(len(display_df)):
                qualified = bool(display_df.iloc[row_idx]["Q"])
                row_color = "#ecfdf5" if qualified else "#f9fafb"
                for col_idx in range(len(display_df.columns)):
                    table[(row_idx + 1, col_idx)].set_facecolor(row_color)
        for ax in axes[len(standings_df["group"].unique()):]:
            ax.axis("off")
        plt.suptitle(f"WM 2026 Group Stage Standings - {method.upper()}", fontsize=18, fontweight="bold", y=0.995)
        plt.tight_layout()
        fig.savefig(group_svg_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        summary_rows.append({"method": method, "champion": champion.winner})
        print(f"Wrote {matches_path}")
        print(f"Wrote {standings_path}")
        print(f"Wrote {svg_path}")
        print(f"Wrote {group_svg_path}")
        print(f"{method.upper()} champion: {champion.winner}")

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "likely_paths_summary.csv", index=False)
    print(f"Wrote {OUTPUT_DIR / 'likely_paths_summary.csv'}")


if __name__ == "__main__":
    main()
