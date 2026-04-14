from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge unrelated lifestyle features into the sleep dataset and min-max scale model inputs."
    )
    parser.add_argument("--base", default="data/all_datasets_model_input.csv")
    parser.add_argument("--extras", default="data/lifestyle_features.csv")
    parser.add_argument("--schema", default="config/feature_schema.json")
    parser.add_argument(
        "--output", default="data/all_datasets_model_input_extended_scaled.csv")
    parser.add_argument(
        "--meta-output", default="outputs/extended_feature_scaling_metadata.json")
    return parser.parse_args()


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _encode_qualitative(series: pd.Series, scale: dict[str, float]) -> pd.Series:
    lookup = {_normalize_key(label): float(score)
              for label, score in scale.items()}

    encoded_values = []
    for raw in series.astype(str):
        text = raw.strip()
        try:
            encoded_values.append(float(text))
            continue
        except ValueError:
            pass
        encoded_values.append(lookup.get(_normalize_key(text), np.nan))

    return pd.Series(encoded_values, index=series.index, dtype=float)


def _encode_categorical(series: pd.Series, mapping: dict[str, float]) -> pd.Series:
    lookup = {_normalize_key(key): float(value)
              for key, value in mapping.items()}
    return pd.Series([
        lookup.get(_normalize_key(str(value)), np.nan)
        for value in series
    ], index=series.index, dtype=float)


def _min_max_scale(series: pd.Series) -> tuple[pd.Series, float, float]:
    min_value = float(series.min(skipna=True))
    max_value = float(series.max(skipna=True))
    if np.isclose(min_value, max_value):
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index), min_value, max_value
    scaled = (series - min_value) / (max_value - min_value)
    return scaled.astype(float), min_value, max_value


def main() -> None:
    args = parse_args()

    base_path = Path(args.base)
    extras_path = Path(args.extras)
    schema_path = Path(args.schema)

    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_path}")
    if not extras_path.exists():
        raise FileNotFoundError(
            f"Lifestyle feature dataset not found: {extras_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    base_df = pd.read_csv(base_path)
    extras_df = pd.read_csv(extras_path)

    if "record_id" not in base_df.columns:
        raise ValueError("Base dataset must include 'record_id'.")
    if "record_id" not in extras_df.columns:
        raise ValueError("Lifestyle feature dataset must include 'record_id'.")

    merged = base_df.merge(extras_df, on="record_id", how="left")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    quantitative_schema = schema["quantitative_features"]
    qualitative_schema = schema["qualitative_features"]

    # Encode model's existing features into numeric form first.
    encoded = pd.DataFrame({"record_id": merged["record_id"]})
    scale_meta: dict[str, dict[str, float]] = {}

    for feature in quantitative_schema:
        encoded[feature] = pd.to_numeric(merged[feature], errors="coerce")

    for feature, config in qualitative_schema.items():
        encoded[feature] = _encode_qualitative(
            merged[feature], config["scale"])

    # Add unrelated lifestyle features.
    expected_extra_columns = [
        "meals_per_day",
        "meal_timing",
        "exercise_days_per_week",
        "exercise_time_of_day",
        "screen_hours_before_bed",
        "caffeine_cups_per_day",
        "stress_1_to_10",
        "sunlight_hours_per_day",
    ]

    missing_extras = [
        column for column in expected_extra_columns if column not in merged.columns]
    if missing_extras:
        raise ValueError(
            "Lifestyle feature dataset is missing required columns: " +
            ", ".join(missing_extras)
        )

    encoded["meals_per_day"] = pd.to_numeric(
        merged["meals_per_day"], errors="coerce")
    encoded["exercise_days_per_week"] = pd.to_numeric(
        merged["exercise_days_per_week"], errors="coerce")
    encoded["screen_hours_before_bed"] = pd.to_numeric(
        merged["screen_hours_before_bed"], errors="coerce")
    encoded["caffeine_cups_per_day"] = pd.to_numeric(
        merged["caffeine_cups_per_day"], errors="coerce")
    encoded["stress_1_to_10"] = pd.to_numeric(
        merged["stress_1_to_10"], errors="coerce")
    encoded["sunlight_hours_per_day"] = pd.to_numeric(
        merged["sunlight_hours_per_day"], errors="coerce")

    encoded["meal_timing"] = _encode_categorical(
        merged["meal_timing"],
        {"early": 1.0, "mid": 2.0, "late": 3.0},
    )
    encoded["exercise_time_of_day"] = _encode_categorical(
        merged["exercise_time_of_day"],
        {"morning": 1.0, "afternoon": 2.0, "evening": 3.0, "night": 4.0},
    )

    # Fill missing values with column median after encoding.
    for column in encoded.columns:
        if column == "record_id":
            continue
        median = float(encoded[column].median(skipna=True)
                       ) if not encoded[column].dropna().empty else 0.0
        encoded[column] = encoded[column].fillna(median)

    # Min-max scale all numeric columns into [0, 1].
    scaled = pd.DataFrame({"record_id": encoded["record_id"]})
    for column in encoded.columns:
        if column == "record_id":
            continue
        scaled_column, min_value, max_value = _min_max_scale(encoded[column])
        scaled[column] = scaled_column
        scale_meta[column] = {"min": min_value, "max": max_value}

    # Sleep quality score using the existing schema weights on original 10 features.
    all_weighted_features = list(
        quantitative_schema.keys()) + list(qualitative_schema.keys())
    weights = np.array(
        [quantitative_schema[name]["weight"] for name in quantitative_schema]
        + [qualitative_schema[name]["weight"] for name in qualitative_schema],
        dtype=float,
    )
    weights = weights / weights.sum()
    scaled["sleep_quality_score"] = scaled[all_weighted_features].to_numpy(
        dtype=float) @ weights

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scaled.to_csv(output_path, index=False)

    meta_output = Path(args.meta_output)
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    meta_output.write_text(
        json.dumps(
            {
                "base_dataset": str(base_path.resolve()),
                "extras_dataset": str(extras_path.resolve()),
                "output_dataset": str(output_path.resolve()),
                "scaling": scale_meta,
                "lifestyle_features": expected_extra_columns,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Extended feature merge + scaling complete.")
    print(f"Rows: {len(scaled)}")
    print(f"Saved scaled dataset to: {output_path.resolve()}")
    print(f"Saved scaling metadata to: {meta_output.resolve()}")


if __name__ == "__main__":
    main()
