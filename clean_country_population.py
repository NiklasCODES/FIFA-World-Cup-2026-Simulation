from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_FILENAME = "country_population.csv"
DEFAULT_INPUT = Path.home() / "Downloads" / RAW_FILENAME
DEFAULT_OUTPUT = BASE_DIR / "data" / "country_population_cleaned.csv"


def strip_html(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_location(value: object) -> str:
    text = strip_html(value)
    text = re.sub(r"\s*\([^)]+\)\s*$", "", text).strip()
    return text


def parse_population(value: object) -> float:
    text = strip_html(value)
    digits = re.sub(r"[^\d]", "", text)
    return float(digits) if digits else float("nan")


def parse_share(value: object) -> float:
    text = strip_html(value)
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    return float(match.group(1)) if match else float("nan")


def parse_date(value: object) -> str:
    text = strip_html(value)
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return ""
    return parsed.date().isoformat()


def clean_population_table(input_path: Path, output_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(input_path, dtype=str)
    cleaned = pd.DataFrame(
        {
            "Location": raw["Location"].map(clean_location),
            "location_clean": raw["Location"].map(clean_location),
            "population": raw["Population"].map(parse_population).astype("Float64"),
            "population_millions": raw["Population"].map(parse_population).astype("Float64") / 1_000_000.0,
            "world_share_pct": raw["% ofmworld"].map(parse_share).astype("Float64"),
            "date_parsed": raw["Date"].map(parse_date),
            "source_clean": raw["Source (official or from the United Nations)"].map(strip_html),
            "notes_clean": raw["Notes"].map(strip_html),
        }
    )
    cleaned = cleaned.sort_values("population", ascending=False, na_position="last").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return cleaned


def main() -> None:
    input_path = DEFAULT_INPUT if DEFAULT_INPUT.exists() else None
    if input_path is None:
        raise FileNotFoundError(f"Could not find raw population CSV at {DEFAULT_INPUT}")
    cleaned = clean_population_table(input_path, DEFAULT_OUTPUT)
    print(f"Wrote {DEFAULT_OUTPUT}")
    print(cleaned.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
