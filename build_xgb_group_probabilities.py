from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from simulate_worldcup_2026 import DEFAULT_GROUP_DRAW_PATH, load_group_draw_from_csv, precompute_matchups, simulate_group_stage
from worldcup_models import prepare_models


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "worldcup_2026_outputs" / "group_probs_xgb_by_group.csv"


def main() -> None:
    models = prepare_models()
    groups = load_group_draw_from_csv(DEFAULT_GROUP_DRAW_PATH, models["alias_map"])
    teams = sorted({team for group in groups.values() for team in group})
    matchups = precompute_matchups(models, teams)
    strength_lookup = models["strength_table"].set_index("team")["combined_strength"].to_dict()

    n_sims = 10000
    rng = np.random.default_rng(42)

    position_counts = {team: Counter() for team in teams}
    qualify_counts = Counter()

    for _ in range(n_sims):
        group_table = simulate_group_stage(rng, "xgb", groups, matchups, strength_lookup)
        for row in group_table.itertuples(index=False):
            position_counts[row.team][int(row.group_rank)] += 1

        qualifiers = group_table[group_table["group_rank"] <= 3].copy()
        qualifiers = qualifiers.sort_values(["group", "group_rank"])
        top_two = qualifiers[qualifiers["group_rank"] <= 2]
        thirds = group_table[group_table["group_rank"] == 3].copy()
        thirds = thirds.sort_values(["points", "gd", "gf", "strength", "team"], ascending=[False, False, False, False, True]).head(8)
        qualify_counts.update(top_two["team"].tolist())
        qualify_counts.update(thirds["team"].tolist())

    rows = []
    for group_name, group_teams in groups.items():
        for team in group_teams:
            rows.append(
                {
                    "group": group_name,
                    "team": team,
                    "group_win_prob": position_counts[team][1] / n_sims,
                    "runner_up_prob": position_counts[team][2] / n_sims,
                    "third_place_prob": position_counts[team][3] / n_sims,
                    "fourth_place_prob": position_counts[team][4] / n_sims,
                    "qualify_r32_prob": qualify_counts[team] / n_sims,
                }
            )

    df = pd.DataFrame(rows).sort_values(
        ["group", "qualify_r32_prob", "group_win_prob", "runner_up_prob", "team"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")
    for group_name, sub in df.groupby("group"):
        print(f"\nGroup {group_name}")
        print(
            sub[[
                "team",
                "group_win_prob",
                "runner_up_prob",
                "third_place_prob",
                "fourth_place_prob",
                "qualify_r32_prob",
            ]].to_string(
                index=False,
                formatters={
                    "group_win_prob": "{:.1%}".format,
                    "runner_up_prob": "{:.1%}".format,
                    "third_place_prob": "{:.1%}".format,
                    "fourth_place_prob": "{:.1%}".format,
                    "qualify_r32_prob": "{:.1%}".format,
                },
            )
        )


if __name__ == "__main__":
    main()
