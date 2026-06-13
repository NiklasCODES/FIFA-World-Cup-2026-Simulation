from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "worldcup_2026_outputs" / "group_probs_xgb_by_group.csv"
OUTPUT_PATH = BASE_DIR / "worldcup_2026_outputs" / "group_probs_xgb_by_group.svg"


def format_pct(value: float) -> str:
    return f"{value:.1%}"


def draw_group_table(ax, group_name: str, sub: pd.DataFrame) -> None:
    ax.axis("off")
    display = sub[[
        "team",
        "group_win_prob",
        "runner_up_prob",
        "third_place_prob",
        "fourth_place_prob",
        "qualify_r32_prob",
    ]].copy()
    display.columns = ["Team", "Win", "R2", "3rd", "4th", "Qualify"]
    for col in display.columns[1:]:
        display[col] = display[col].map(format_pct)

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.05, 1.55)

    # Header styling
    for col_idx in range(len(display.columns)):
        cell = table[(0, col_idx)]
        cell.set_facecolor("#111827")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")

    # Body styling
    for row_idx in range(len(display)):
        qualify = float(sub.iloc[row_idx]["qualify_r32_prob"])
        if qualify >= 0.9:
            row_color = "#ecfdf5"
        elif qualify >= 0.75:
            row_color = "#f0f9ff"
        elif qualify >= 0.5:
            row_color = "#fff7ed"
        else:
            row_color = "#fef2f2"
        for col_idx in range(len(display.columns)):
            cell = table[(row_idx + 1, col_idx)]
            cell.set_facecolor(row_color)
            if col_idx == 0:
                cell.get_text().set_ha("left")
                cell.PAD = 0.12

    ax.set_title(f"Group {group_name}", fontsize=13, fontweight="bold", pad=10)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    groups = sorted(df["group"].unique())
    fig, axes = plt.subplots(4, 3, figsize=(22, 24))
    axes = axes.flatten()

    for ax, group_name in zip(axes, groups):
        sub = df[df["group"] == group_name].sort_values(
            ["qualify_r32_prob", "group_win_prob", "runner_up_prob", "team"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        draw_group_table(ax, group_name, sub)

    for ax in axes[len(groups):]:
        ax.axis("off")

    fig.suptitle(
        "FIFA World Cup 2026 - XGBoost Group Stage Probabilities",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
