from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STRESS_MAP = {
    "not_at_all": 2.0,
    "a_little": 4.0,
    "moderately": 6.0,
    "very": 8.0,
    "extremely": 9.5,
}

SLEEPINESS_MAP = {
    "never": 1.0,
    "rarely": 2.0,
    "sometimes": 3.5,
    "often": 5.0,
    "constantly": 6.0,
}

SCREEN_MAP = {
    "none": 0.2,
    "brief": 0.8,
    "moderate": 1.8,
    "heavy": 3.0,
    "very_heavy": 4.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-fill lifestyle features heuristically from existing sleep records."
    )
    parser.add_argument("--base", default="data/all_datasets_model_input.csv")
    parser.add_argument("--output", default="data/lifestyle_features.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(series.median(numeric_only=False) if hasattr(series, "median") else 0.0)


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_path}")

    df = pd.read_csv(base_path)
    if "record_id" not in df.columns:
        raise ValueError("Base dataset must include 'record_id'.")

    rng = np.random.default_rng(args.seed)

    sleep_hours = pd.to_numeric(df.get("total_sleep_hours", 7.0), errors="coerce").fillna(7.0)
    efficiency = pd.to_numeric(df.get("sleep_efficiency_pct", 80.0), errors="coerce").fillna(80.0)
    interruptions = pd.to_numeric(df.get("interruptions", 2.0), errors="coerce").fillna(2.0)

    perceived_stress = df.get("perceived_stress", pd.Series(["moderately"] * len(df))).astype(str).str.strip().str.lower()
    daytime_sleepiness = df.get("daytime_sleepiness", pd.Series(["sometimes"] * len(df))).astype(str).str.strip().str.lower()
    screen_before_bed = df.get("screen_time_before_bed", pd.Series(["moderate"] * len(df))).astype(str).str.strip().str.lower()

    stress_base = perceived_stress.map(STRESS_MAP).fillna(6.0)
    sleepiness_base = daytime_sleepiness.map(SLEEPINESS_MAP).fillna(3.0)
    screen_base = screen_before_bed.map(SCREEN_MAP).fillna(1.8)

    meals_per_day = np.clip(np.round(3.2 - interruptions / 6.0 + rng.normal(0.0, 0.6, len(df))), 1, 6).astype(int)

    meal_late_signal = (screen_base >= 2.8) | (sleep_hours < 6.5)
    meal_early_signal = (sleep_hours > 8.0) & (efficiency > 88.0)
    meal_timing = np.where(meal_late_signal, "late", np.where(meal_early_signal, "early", "mid"))

    exercise_days = np.clip(
        np.round(4.5 + (efficiency - 80.0) / 8.0 - interruptions / 8.0 + rng.normal(0.0, 1.2, len(df))),
        0,
        7,
    ).astype(int)

    ex_time_draw = rng.random(len(df))
    exercise_time = np.where(ex_time_draw < 0.35, "morning", np.where(ex_time_draw < 0.65, "afternoon", "evening"))
    exercise_time = np.where(exercise_days <= 1, "night", exercise_time)

    screen_hours_before_bed = np.clip(screen_base + rng.normal(0.0, 0.35, len(df)), 0.0, 6.0)
    caffeine_cups_per_day = np.clip(1.0 + sleepiness_base * 0.7 + interruptions * 0.2 + rng.normal(0.0, 0.7, len(df)), 0.0, 8.0)
    stress_1_to_10 = np.clip(stress_base + rng.normal(0.0, 0.9, len(df)), 1.0, 10.0)
    sunlight_hours = np.clip(5.0 + (sleep_hours - 7.0) * 0.8 - stress_1_to_10 * 0.2 + rng.normal(0.0, 1.0, len(df)), 0.0, 12.0)

    out = pd.DataFrame(
        {
            "record_id": df["record_id"],
            "meals_per_day": meals_per_day,
            "meal_timing": meal_timing,
            "exercise_days_per_week": exercise_days,
            "exercise_time_of_day": exercise_time,
            "screen_hours_before_bed": np.round(screen_hours_before_bed, 2),
            "caffeine_cups_per_day": np.round(caffeine_cups_per_day, 2),
            "stress_1_to_10": np.round(stress_1_to_10, 2),
            "sunlight_hours_per_day": np.round(sunlight_hours, 2),
        }
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print("Lifestyle features auto-filled.")
    print(f"Rows: {len(out)}")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()