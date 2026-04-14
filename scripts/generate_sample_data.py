from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "sample_sleep_data.csv"


def bucketize(value: float, labels: list[str]) -> str:
    index = int(np.clip(round(value), 1, len(labels))) - 1
    return labels[index]


def main() -> None:
    rng = np.random.default_rng(42)
    row_count = 240

    stress_labels = ["not_at_all", "a_little", "moderately", "very", "extremely"]
    mood_labels = ["awful", "poor", "okay", "good", "excellent"]
    sleepiness_labels = ["never", "rarely", "sometimes", "often", "constantly"]
    screen_labels = ["none", "brief", "moderate", "heavy", "very_heavy"]

    fieldnames = [
        "record_id",
        "total_sleep_hours",
        "sleep_efficiency_pct",
        "interruptions",
        "sleep_onset_latency_min",
        "wake_variability_min",
        "bedtime_variability_min",
        "perceived_stress",
        "morning_mood",
        "daytime_sleepiness",
        "screen_time_before_bed",
    ]

    rows: list[dict[str, str | float | int]] = []
    latent_quality = np.clip(rng.beta(2.8, 2.3, size=row_count), 0.0, 1.0)

    for index, quality in enumerate(latent_quality, start=1):
        total_sleep_hours = np.clip(4.5 + 3.8 * quality + rng.normal(0.0, 0.55), 3.2, 9.5)
        sleep_efficiency_pct = np.clip(67 + 28 * quality + rng.normal(0.0, 4.5), 60, 99)
        interruptions = int(np.clip(round(5.0 - 4.2 * quality + rng.normal(0.0, 0.85)), 0, 8))
        sleep_onset_latency_min = np.clip(14 + 52 * (1.0 - quality) + rng.normal(0.0, 7.5), 0, 90)
        wake_variability_min = np.clip(18 + 120 * (1.0 - quality) + rng.normal(0.0, 12.0), 0, 180)
        bedtime_variability_min = np.clip(12 + 130 * (1.0 - quality) + rng.normal(0.0, 14.0), 0, 180)

        perceived_stress = bucketize(1 + 4 * (1.0 - quality) + rng.normal(0.0, 0.55), stress_labels)
        morning_mood = bucketize(1 + 4 * quality + rng.normal(0.0, 0.5), mood_labels)
        daytime_sleepiness = bucketize(1 + 4 * (1.0 - quality) + rng.normal(0.0, 0.55), sleepiness_labels)
        screen_time_before_bed = bucketize(1 + 3.2 * (1.0 - quality) + rng.normal(0.0, 0.85), screen_labels)

        rows.append(
            {
                "record_id": f"night_{index:03d}",
                "total_sleep_hours": round(float(total_sleep_hours), 2),
                "sleep_efficiency_pct": round(float(sleep_efficiency_pct), 2),
                "interruptions": interruptions,
                "sleep_onset_latency_min": round(float(sleep_onset_latency_min), 2),
                "wake_variability_min": round(float(wake_variability_min), 2),
                "bedtime_variability_min": round(float(bedtime_variability_min), 2),
                "perceived_stress": perceived_stress,
                "morning_mood": morning_mood,
                "daytime_sleepiness": daytime_sleepiness,
                "screen_time_before_bed": screen_time_before_bed,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote sample dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
