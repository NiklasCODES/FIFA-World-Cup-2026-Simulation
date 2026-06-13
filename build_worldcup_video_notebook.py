from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUTPUT = Path("worldcup_video_models.ipynb")


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
            "# WM 2026 Forecasting Notebook\n"
            "Dieses Notebook zeigt eine reproduzierbare Analyse fuer dein Video: Datenqualitaet, probabilistische Modelle, XGBoost und eine Demo-Knockout-Simulation auf Basis der staerksten Teams.\n"
            "\n"
            "Hinweis: Die Turnierdaten enthalten keine offizielle 48er-WM-Gruppenphase, daher nutzt die Simulation einen demonstrativen 16er-Knockout nach Teamstaerke."
        ),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from IPython.display import display\n"
            "\n"
            "from worldcup_models import (\n"
            "    prepare_models,\n"
            "    build_match_context,\n"
            "    predict_matchup,\n"
            "    demo_bracket_teams,\n"
            "    simulate_knockout_bracket,\n"
            "    load_sources,\n"
            ")\n"
            "\n"
            "sns.set_theme(style='whitegrid', context='talk')\n"
            "plt.rcParams['figure.figsize'] = (12, 6)\n"
            "pd.set_option('display.max_columns', 200)\n"
            "pd.set_option('display.width', 160)\n"
        ),
        md("## 1. Datenqualitaet\nHier pruefen wir die FC-26-Spielerdaten auf fehlende Werte, Typprobleme und auffaellige Felder."),
        code(
            "games, teams, players = load_sources()\n"
            "fc = players.copy()\n"
            "\n"
            "missing = (\n"
            "    fc.isna().mean().sort_values(ascending=False).head(15).reset_index()\n"
            ")\n"
            "missing.columns = ['Spalte', 'Missing-Rate']\n"
            "display(missing.style.format({'Missing-Rate': '{:.1%}'}))\n"
            "\n"
            "fig, ax = plt.subplots()\n"
            "sns.barplot(data=missing, y='Spalte', x='Missing-Rate', ax=ax, color='#1f77b4')\n"
            "ax.set_title('Top-15 Missing-Rate in FC26_20250921.csv')\n"
            "ax.set_xlabel('Missing-Rate')\n"
            "ax.set_ylabel('')\n"
            "plt.tight_layout()\n"
        ),
        code(
            "quality_checks = pd.DataFrame([\n"
            "    {'Check': 'Doppelte Spalten', 'Wert': int(fc.columns.duplicated().sum())},\n"
            "    {'Check': 'Doppelte player_id', 'Wert': int(fc['player_id'].duplicated().sum())},\n"
            "    {'Check': 'All-NaN-Spalten', 'Wert': int(sum(fc[c].isna().all() for c in fc.columns))},\n"
            "])\n"
            "display(quality_checks)\n"
            "display(fc[['overall', 'potential', 'age', 'value_eur', 'wage_eur']].describe().T)\n"
        ),
        md("## 2. Modelle trainieren\nWir kombinieren ein Poisson-Modell fuer Tore, ein logistisches Modell fuer 1X2 und ein XGBoost-Modell fuer dieselbe Zielvariable."),
        code(
            "models = prepare_models()\n"
            "metrics = pd.DataFrame(models['train_metrics'].items(), columns=['Metrik', 'Wert'])\n"
            "display(metrics.style.format({'Wert': '{:.4f}'}))\n"
            "\n"
            "fig, ax = plt.subplots()\n"
            "sns.barplot(data=metrics, x='Wert', y='Metrik', ax=ax, palette='viridis')\n"
            "ax.set_title('Modellvergleich auf historischem Holdout')\n"
            "ax.set_xlabel('Score')\n"
            "ax.set_ylabel('')\n"
            "plt.tight_layout()\n"
        ),
        md("## 3. Staerkste Teams\nDie Teamstaerke basiert auf FC26-Ratings, Positionstiefe und den historischen Team-Metadaten."),
        code(
            "strength = models['strength_table'][['team', 'combined_strength', 'fifa_ranking', 'fc26_top11_overall_mean']].head(15).copy()\n"
            "display(strength.style.format({'combined_strength': '{:.2f}', 'fifa_ranking': '{:.0f}', 'fc26_top11_overall_mean': '{:.2f}'}))\n"
            "\n"
            "fig, ax = plt.subplots()\n"
            "sns.barplot(data=strength, y='team', x='combined_strength', ax=ax, color='#d62728')\n"
            "ax.set_title('Top 15 Teams nach kombinierter Staerke')\n"
            "ax.set_xlabel('Combined Strength')\n"
            "ax.set_ylabel('')\n"
            "plt.tight_layout()\n"
        ),
        md("## 4. Beispiel-Matchup\nHier zeigen wir, wie sich unterschiedliche Modelle fuer ein konkretes Spiel unterscheiden."),
        code(
            "def matchup_table(home: str, away: str) -> pd.DataFrame:\n"
            "    row = build_match_context(models, home, away)\n"
            "    rows = []\n"
            "    for method in ['poisson', 'logistic', 'xgb']:\n"
            "        pred = predict_matchup(models, row, method=method)\n"
            "        rows.append({\n"
            "            'Modell': method.upper(),\n"
            "            'Heimsieg': pred['prob_home_win'],\n"
            "            'Remis': pred['prob_draw'],\n"
            "            'Auswaertssieg': pred['prob_away_win'],\n"
            "            'Ert. Heimtore': np.nan if pred['expected_home_goals'] is None else pred['expected_home_goals'],\n"
            "            'Ert. Auswaertstore': np.nan if pred['expected_away_goals'] is None else pred['expected_away_goals'],\n"
            "        })\n"
            "    return pd.DataFrame(rows)\n"
            "\n"
            "example = matchup_table('France', 'Spain')\n"
            "display(example.style.format({'Heimsieg': '{:.1%}', 'Remis': '{:.1%}', 'Auswaertssieg': '{:.1%}', 'Ert. Heimtore': '{:.2f}', 'Ert. Auswaertstore': '{:.2f}'}))\n"
            "\n"
            "plot_df = example.melt(id_vars='Modell', value_vars=['Heimsieg', 'Remis', 'Auswaertssieg'], var_name='Ausgang', value_name='Wahrscheinlichkeit')\n"
            "fig, ax = plt.subplots()\n"
            "sns.barplot(data=plot_df, x='Modell', y='Wahrscheinlichkeit', hue='Ausgang', ax=ax)\n"
            "ax.set_title('France vs Spain: Ergebnis-Wahrscheinlichkeiten')\n"
            "ax.set_ylim(0, 1)\n"
            "plt.tight_layout()\n"
        ),
        md("## 5. Demo-Knockout\nWir simulieren die K.o.-Phase mit den staerksten 16 Teams als fuer die Videoerzaehlung brauchbare Demo."),
        code(
            "bracket = demo_bracket_teams(models['strength_table'], size=16)\n"
            "print('Demo bracket:', ', '.join(bracket))\n"
            "\n"
            "def predictor(home: str, away: str):\n"
            "    row = build_match_context(models, home, away)\n"
            "    return predict_matchup(models, row, method='xgb')\n"
            "\n"
            "champions = simulate_knockout_bracket(bracket, predictor, n_simulations=3000)\n"
            "display(champions.head(10).style.format({'champion_probability': '{:.1%}'}))\n"
            "\n"
            "fig, ax = plt.subplots()\n"
            "sns.barplot(data=champions.head(10), y='team', x='champion_probability', ax=ax, color='#2ca02c')\n"
            "ax.set_title('Top 10 Champion-Wahrscheinlichkeiten (XGBoost, Demo-Knockout)')\n"
            "ax.set_xlabel('Champion-Wahrscheinlichkeit')\n"
            "ax.set_ylabel('')\n"
            "from matplotlib.ticker import FuncFormatter\n"
            "ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0%}'))\n"
            "plt.tight_layout()\n"
        ),
        md("## 6. Naechster Schritt\nWenn du willst, kann ich dir als naechstes noch eine kompakte `README` oder eine YouTube-Skriptstruktur aus dieser Notebook-Story bauen."),
    ]

    nb["cells"] = cells
    return nb


def main() -> None:
    nb = build_notebook()
    OUTPUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
