from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUTPUT = Path("worldcup_2026_report.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(source: str):
    return nbf.v4.new_code_cell(source)


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

    cells = [
        md(
            "# World Cup 2026 Simulation Report\n"
            "Dieses Notebook zeigt die 10.000er-Monte-Carlo-Auswertung fuer `Poisson`, `Logistic` und `XGBoost`.\n"
            "Die Charts und Tabellen sind auf Video-Use ausgelegt: erst Draw, dann Modellvergleich, dann Champions und Turnierpfad."
        ),
        code(
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from pathlib import Path\n"
            "from IPython.display import display\n"
            "from matplotlib.ticker import PercentFormatter\n"
            "\n"
            "sns.set_theme(style='whitegrid', context='talk')\n"
            "plt.rcParams['figure.figsize'] = (14, 7)\n"
            "pd.set_option('display.max_columns', 200)\n"
            "pd.set_option('display.width', 180)\n"
            "\n"
            "base = 'worldcup_2026_outputs'\n"
            "group_draw = pd.read_csv('data/group_draw_2026.csv') if Path('data/group_draw_2026.csv').exists() else pd.read_csv(f'{base}/group_draw.csv')\n"
            "champions = {method: pd.read_csv(f'{base}/champions_{method}.csv') for method in ['poisson', 'logistic', 'xgb']}\n"
            "stage_probs = {method: pd.read_csv(f'{base}/stage_probs_{method}.csv') for method in ['poisson', 'logistic', 'xgb']}\n"
        ),
        md("## 1. Gruppen-Auslosung"),
        code(
            "draw_pivot = group_draw.pivot(index='group', columns='position', values='team')\n"
            "draw_pivot.columns = ['Pot-/Platz 1', 'Pot-/Platz 2', 'Pot-/Platz 3', 'Pot-/Platz 4']\n"
            "display(draw_pivot)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(14, 8))\n"
            "ax.axis('off')\n"
            "table = ax.table(cellText=draw_pivot.values, rowLabels=draw_pivot.index, colLabels=draw_pivot.columns, loc='center')\n"
            "table.auto_set_font_size(False)\n"
            "table.set_fontsize(11)\n"
            "table.scale(1, 1.6)\n"
            "ax.set_title('WM 2026 Gruppenauslosung', pad=20)\n"
            "plt.tight_layout()\n"
        ),
        md("## 1b. Gruppentabellen"),
        code(
            "group_rank_table = pd.read_csv(f'{base}/featured_group_standings.csv')\n"
            "display(group_rank_table)\n"
            "\n"
            "fig, axes = plt.subplots(4, 3, figsize=(18, 20))\n"
            "axes = axes.flatten()\n"
            "for ax, group in zip(axes, sorted(group_rank_table['group'].unique())):\n"
            "    ax.axis('off')\n"
            "    sub = group_rank_table[group_rank_table['group'] == group].sort_values(['group_rank', 'points', 'gd', 'gf'], ascending=[True, False, False, False])\n"
            "    show = sub[['group_rank', 'team', 'points', 'gd', 'gf', 'qualified']].copy()\n"
            "    show.columns = ['Rank', 'Team', 'Pts', 'GD', 'GF', 'Q']\n"
            "    tbl = ax.table(cellText=show.values, colLabels=show.columns, cellLoc='center', loc='center')\n"
            "    tbl.auto_set_font_size(False)\n"
            "    tbl.set_fontsize(8)\n"
            "    tbl.scale(1, 1.35)\n"
            "    ax.set_title(f'Group {group}', fontsize=12, fontweight='bold')\n"
            "plt.tight_layout()\n"
        ),
        md("## 2. Champions nach Modell"),
        code(
            "summary = []\n"
            "for method, df in champions.items():\n"
            "    top = df.head(10).copy()\n"
            "    top['model'] = method.upper()\n"
            "    summary.append(top)\n"
            "champion_summary = pd.concat(summary, ignore_index=True)\n"
            "display(champion_summary)\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharex=False)\n"
            "for ax, (method, df) in zip(axes, champions.items()):\n"
            "    top = df.head(10).sort_values('champion_prob', ascending=True)\n"
            "    sns.barplot(data=top, y='team', x='champion_prob', ax=ax, color='#1f77b4')\n"
            "    ax.set_title(method.upper())\n"
            "    ax.set_xlabel('Champion-Wahrscheinlichkeit')\n"
            "    ax.set_ylabel('')\n"
            "    ax.xaxis.set_major_formatter(PercentFormatter(1.0))\n"
            "plt.suptitle('Top-10 Champion-Wahrscheinlichkeiten je Modell', y=1.02)\n"
            "plt.tight_layout()\n"
        ),
        md("## 3. Bracket"),
        code(
            "from IPython.display import SVG, display\n"
            "display(SVG(f'{base}/featured_bracket.svg'))\n"
            "\n"
            "bracket_df = pd.read_csv(f'{base}/featured_bracket_table.csv')\n"
            "display(bracket_df.head(20))\n"
        ),
        md("## 4. Final- und Halbfinalchancen"),
        code(
            "stage_tables = []\n"
            "for method, df in stage_probs.items():\n"
            "    temp = df.copy()\n"
            "    temp['model'] = method.upper()\n"
            "    stage_tables.append(temp)\n"
            "stage_all = pd.concat(stage_tables, ignore_index=True)\n"
            "display(stage_all.head(30))\n"
            "\n"
            "top_for_heatmap = stage_all.sort_values(['model', 'champion_prob'], ascending=[True, False]).groupby('model').head(12)\n"
            "heat = top_for_heatmap.pivot(index='team', columns='model', values='champion_prob').fillna(0)\n"
            "fig, ax = plt.subplots(figsize=(12, 8))\n"
            "sns.heatmap(heat, annot=True, fmt='.0%', cmap='Blues', ax=ax)\n"
            "ax.set_title('Champion-Wahrscheinlichkeiten der Top-Teams')\n"
            "plt.tight_layout()\n"
        ),
        md("## 5. Video-taugliche Kurzfassung"),
        code(
            "for method, df in champions.items():\n"
            "    top = df.head(5).copy()\n"
            "    top['champion_prob'] = top['champion_prob'].map(lambda x: f'{x:.1%}')\n"
            "    print(f'\\n{method.upper()}')\n"
            "    print(top.to_string(index=False))\n"
        ),
    ]

    nb["cells"] = cells
    return nb


def main() -> None:
    nb = build_notebook()
    OUTPUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
