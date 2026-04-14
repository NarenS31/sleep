from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a template CSV for unrelated lifestyle features keyed by record_id."
    )
    parser.add_argument("--base", default="data/all_datasets_model_input.csv")
    parser.add_argument("--output", default="data/lifestyle_features.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_path}")

    base = pd.read_csv(base_path)
    if "record_id" not in base.columns:
        raise ValueError("Base dataset must include 'record_id'.")

    template = pd.DataFrame(
        {
            "record_id": base["record_id"],
            "meals_per_day": "",
            "meal_timing": "",
            "exercise_days_per_week": "",
            "exercise_time_of_day": "",
            "screen_hours_before_bed": "",
            "caffeine_cups_per_day": "",
            "stress_1_to_10": "",
            "sunlight_hours_per_day": "",
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)

    print("Lifestyle feature template created.")
    print(f"Rows: {len(template)}")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
