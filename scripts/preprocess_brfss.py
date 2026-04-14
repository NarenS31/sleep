from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BRFSS_MISSING_CODES = {
    7,
    9,
    77,
    88,
    99,
    777,
    888,
    999,
    7777,
    8888,
    9999,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BRFSS data into the project's model input CSV format.")
    parser.add_argument("--input", required=True,
                        help="Path to BRFSS data file (.xpt, .csv, .txt).")
    parser.add_argument(
        "--output", default="data/brfss_model_input.csv", help="Output CSV path.")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Optional cap for output row count (0 means no cap).")
    return parser.parse_args()


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    return frame


def _first_available(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _to_numeric(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.mask(numeric.isin(BRFSS_MISSING_CODES))
    return numeric


def _bin_label(values: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    bins = [-np.inf] + edges + [np.inf]
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=True).astype("string")


def _safe_clip(series: pd.Series, minimum: float, maximum: float) -> pd.Series:
    return series.clip(lower=minimum, upper=maximum)


def load_brfss(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xpt":
        frame = pd.read_sas(path, format="xport")
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".txt", ".asc", ".dat"}:
        frame = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Use .xpt or .csv.")
    return _normalize_columns(frame)


def build_model_frame(brfss: pd.DataFrame) -> pd.DataFrame:
    columns = list(brfss.columns)

    sleep_hours_col = _first_available(
        columns, ["SLEPTIM1", "SLEPHRS", "SLEEPHRS"])
    mental_days_col = _first_available(columns, ["MENTHLTH", "_MENT14D"])
    general_health_col = _first_available(columns, ["GENHLTH"])
    physical_days_col = _first_available(columns, ["PHYSHLTH", "_PHYS14D"])

    stress_days = _to_numeric(
        brfss[mental_days_col]) if mental_days_col else None
    general_health = _to_numeric(
        brfss[general_health_col]) if general_health_col else None
    physical_days = _to_numeric(
        brfss[physical_days_col]) if physical_days_col else None

    if sleep_hours_col is not None:
        sleep_hours = _to_numeric(brfss[sleep_hours_col])
    else:
        if general_health is None and stress_days is None and physical_days is None:
            raise ValueError(
                "Could not find direct sleep hours or fallback health columns to derive a proxy sleep-hours feature."
            )
        sleep_hours = pd.Series(8.0, index=brfss.index, dtype=float)
        if general_health is not None:
            # GENHLTH is 1=excellent to 5=poor, so higher values reduce proxy sleep duration.
            sleep_hours = sleep_hours - \
                ((general_health.fillna(3.0) - 3.0) * 0.6)
        if stress_days is not None:
            sleep_hours = sleep_hours - (stress_days.fillna(10.0) / 20.0)
        if physical_days is not None:
            sleep_hours = sleep_hours - (physical_days.fillna(10.0) / 25.0)
        sleep_hours = _safe_clip(sleep_hours, 3.0, 10.0)

    frame = pd.DataFrame()
    frame["total_sleep_hours"] = _safe_clip(sleep_hours, 3.0, 10.0)

    sleep_efficiency_proxy = 100.0 - \
        (np.abs(frame["total_sleep_hours"] - 8.0) * 9.0)
    if stress_days is not None:
        sleep_efficiency_proxy = sleep_efficiency_proxy - \
            (stress_days.fillna(stress_days.median()) * 0.25)
    frame["sleep_efficiency_pct"] = _safe_clip(
        sleep_efficiency_proxy, 60.0, 100.0)

    interruptions_proxy = np.maximum(0.0, np.round(
        (8.0 - frame["total_sleep_hours"]).clip(lower=0.0) * 1.4))
    if physical_days is not None:
        interruptions_proxy = interruptions_proxy + \
            np.round(physical_days.fillna(physical_days.median()) / 15.0)
    frame["interruptions"] = _safe_clip(interruptions_proxy, 0.0, 8.0)

    sleep_latency_proxy = (
        8.0 - frame["total_sleep_hours"]).clip(lower=0.0) * 12.0
    if stress_days is not None:
        sleep_latency_proxy = sleep_latency_proxy + \
            stress_days.fillna(stress_days.median()) * 0.7
    frame["sleep_onset_latency_min"] = _safe_clip(
        sleep_latency_proxy, 0.0, 90.0)

    wake_var_proxy = (8.0 - frame["total_sleep_hours"]).abs() * 18.0
    bedtime_var_proxy = (8.0 - frame["total_sleep_hours"]).abs() * 20.0
    if stress_days is not None:
        wake_var_proxy = wake_var_proxy + \
            stress_days.fillna(stress_days.median()) * 1.2
        bedtime_var_proxy = bedtime_var_proxy + \
            stress_days.fillna(stress_days.median()) * 1.4
    frame["wake_variability_min"] = _safe_clip(wake_var_proxy, 0.0, 180.0)
    frame["bedtime_variability_min"] = _safe_clip(
        bedtime_var_proxy, 0.0, 180.0)

    if stress_days is None:
        perceived_stress = _bin_label(
            (8.0 - frame["total_sleep_hours"]).clip(lower=0.0),
            [0.4, 1.0, 1.6, 2.4],
            ["not_at_all", "a_little", "moderately", "very", "extremely"],
        )
    else:
        perceived_stress = _bin_label(
            stress_days,
            [2, 7, 14, 21],
            ["not_at_all", "a_little", "moderately", "very", "extremely"],
        )
    frame["perceived_stress"] = perceived_stress

    if general_health is None:
        mood_score = _safe_clip(
            np.round(frame["total_sleep_hours"] - 3.0), 1.0, 5.0)
    else:
        # BRFSS GENHLTH is 1=excellent ... 5=poor, invert to our positive mood scale.
        mood_score = _safe_clip(6.0 - general_health, 1.0, 5.0)
    frame["morning_mood"] = mood_score.map(
        {
            1.0: "awful",
            2.0: "poor",
            3.0: "okay",
            4.0: "good",
            5.0: "excellent",
        }
    ).astype("string")

    frame["daytime_sleepiness"] = _bin_label(
        (8.0 - frame["total_sleep_hours"]).clip(lower=0.0),
        [0.3, 0.9, 1.6, 2.4],
        ["never", "rarely", "sometimes", "often", "constantly"],
    )

    screen_proxy = 8.0 - frame["total_sleep_hours"]
    if stress_days is not None:
        screen_proxy = screen_proxy + \
            stress_days.fillna(stress_days.median()) / 10.0
    frame["screen_time_before_bed"] = _bin_label(
        screen_proxy,
        [0.4, 1.0, 1.8, 2.6],
        ["none", "brief", "moderate", "heavy", "very_heavy"],
    )

    frame = frame.dropna()
    frame = frame.reset_index(drop=True)
    frame.insert(0, "record_id", [
                 f"brfss_{index + 1:06d}" for index in range(len(frame))])
    return frame


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    brfss = load_brfss(input_path)
    model_frame = build_model_frame(brfss)
    if args.max_rows > 0:
        model_frame = model_frame.iloc[: args.max_rows].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_frame.to_csv(output_path, index=False)

    print("BRFSS preprocessing complete.")
    print(f"Input rows: {len(brfss)}")
    print(f"Output rows: {len(model_frame)}")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
