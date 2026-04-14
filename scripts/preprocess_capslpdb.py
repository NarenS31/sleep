from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SLEEP_EVENTS = {"SLEEP-S1", "SLEEP-S2", "SLEEP-S3", "SLEEP-S4", "SLEEP-REM"}
IN_BED_EVENTS = SLEEP_EVENTS | {"SLEEP-S0"}
LINE_PATTERN = re.compile(
    r"^(?P<stage>\S+)\s+.+?\s+(?P<clock>\d{2}:\d{2}:\d{2})\s+(?P<event>\S+)\s+(?P<duration>\d+)"
)


@dataclass
class Event:
    onset_sec: float
    duration_sec: float
    event_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CAP Sleep DB annotation TXT files into model input CSV format."
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw/fast_s3/capslpdb",
        help="Path to CAP Sleep DB directory.",
    )
    parser.add_argument(
        "--output",
        default="data/capslpdb_model_input.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _bin_label(value: float, edges: list[float], labels: list[str]) -> str:
    bins = [-math.inf] + edges + [math.inf]
    return str(pd.cut(pd.Series([value]), bins=bins, labels=labels, include_lowest=True).iloc[0])


def _derive_qualitative(total_sleep_hours: float, sleep_efficiency_pct: float, interruptions: float, sleep_onset_latency_min: float, bedtime_min: float) -> dict[str, str]:
    stress_score = (8.0 - total_sleep_hours) + (100.0 -
                                                sleep_efficiency_pct) / 20.0 + interruptions * 0.25
    mood_score = 5.3 - (8.0 - total_sleep_hours) - \
        (100.0 - sleep_efficiency_pct) / 17.0
    sleepy_score = (8.0 - total_sleep_hours) + \
        sleep_onset_latency_min / 35.0 + interruptions * 0.1
    screen_proxy = max(0.0, (bedtime_min - 22.5 * 60.0) /
                       35.0) + sleep_onset_latency_min / 42.0

    return {
        "perceived_stress": _bin_label(
            stress_score,
            [0.8, 1.6, 2.5, 3.5],
            ["not_at_all", "a_little", "moderately", "very", "extremely"],
        ),
        "morning_mood": _bin_label(
            mood_score,
            [1.5, 2.5, 3.5, 4.5],
            ["awful", "poor", "okay", "good", "excellent"],
        ),
        "daytime_sleepiness": _bin_label(
            sleepy_score,
            [0.6, 1.3, 2.0, 2.8],
            ["never", "rarely", "sometimes", "often", "constantly"],
        ),
        "screen_time_before_bed": _bin_label(
            screen_proxy,
            [0.4, 1.0, 1.8, 2.6],
            ["none", "brief", "moderate", "heavy", "very_heavy"],
        ),
    }


def _clock_to_seconds(clock_text: str) -> int:
    hh, mm, ss = [int(token) for token in clock_text.split(":")]
    return hh * 3600 + mm * 60 + ss


def _parse_events(path: Path) -> list[Event]:
    events: list[Event] = []
    lines = path.read_text(errors="ignore").splitlines()

    start_index = None
    for index, line in enumerate(lines):
        if line.strip().startswith("Sleep Stage"):
            start_index = index + 1
            break
    if start_index is None:
        return events

    raw_events: list[tuple[int, str, int]] = []
    for line in lines[start_index:]:
        text = line.strip()
        if not text:
            continue
        match = LINE_PATTERN.match(text)
        if not match:
            continue
        event_name = match.group("event").strip().upper()
        if event_name not in IN_BED_EVENTS:
            continue
        raw_events.append(
            (
                _clock_to_seconds(match.group("clock")),
                event_name,
                int(match.group("duration")),
            )
        )

    if not raw_events:
        return events

    base_time = raw_events[0][0]
    rollover = 0
    previous_clock = base_time
    for clock_value, event_name, duration in raw_events:
        if clock_value < previous_clock:
            rollover += 24 * 3600
        absolute = clock_value + rollover
        onset = float(absolute - base_time)
        events.append(Event(onset_sec=onset, duration_sec=float(
            duration), event_name=event_name))
        previous_clock = clock_value

    return events


def _compute_features(events: list[Event], start_clock_sec: int) -> dict[str, float]:
    if not events:
        raise ValueError("No sleep-stage events found.")

    events = sorted(events, key=lambda item: item.onset_sec)
    in_bed_duration = sum(
        event.duration_sec for event in events if event.event_name in IN_BED_EVENTS)
    sleep_duration = sum(
        event.duration_sec for event in events if event.event_name in SLEEP_EVENTS)

    first_sleep = next(
        (event.onset_sec for event in events if event.event_name in SLEEP_EVENTS), 0.0)
    last_sleep_end = max(
        (event.onset_sec + event.duration_sec for event in events if event.event_name in SLEEP_EVENTS),
        default=0.0,
    )

    interruption_count = 0
    in_wake_bout = False
    for event in events:
        if event.onset_sec < first_sleep or event.onset_sec >= last_sleep_end:
            continue
        if event.event_name == "SLEEP-S0" and event.duration_sec >= 30:
            if not in_wake_bout:
                interruption_count += 1
                in_wake_bout = True
        elif event.event_name in SLEEP_EVENTS:
            in_wake_bout = False

    bedtime_min = (start_clock_sec / 60.0) % (24.0 * 60.0)
    wake_min = (start_clock_sec / 60.0 + max(event.onset_sec +
                event.duration_sec for event in events) / 60.0) % (24.0 * 60.0)

    total_sleep_hours = _clip(sleep_duration / 3600.0, 3.0, 10.0)
    sleep_efficiency_pct = _clip(
        (sleep_duration / in_bed_duration) * 100.0 if in_bed_duration > 0 else 60.0, 60.0, 100.0)
    interruptions = _clip(float(interruption_count), 0.0, 8.0)
    sleep_onset_latency_min = _clip(first_sleep / 60.0, 0.0, 90.0)

    # CAP provides mostly one-night records; use within-night wake/sleep fragmentation as a proxy variability term.
    wake_variability_min = _clip(
        interruptions * 8.0 + sleep_onset_latency_min * 0.5, 0.0, 180.0)
    bedtime_variability_min = _clip(
        abs(total_sleep_hours - 7.5) * 25.0 + interruptions * 6.0, 0.0, 180.0)

    return {
        "total_sleep_hours": total_sleep_hours,
        "sleep_efficiency_pct": sleep_efficiency_pct,
        "interruptions": interruptions,
        "sleep_onset_latency_min": sleep_onset_latency_min,
        "wake_variability_min": wake_variability_min,
        "bedtime_variability_min": bedtime_variability_min,
        "bedtime_clock_min": bedtime_min,
        "wake_clock_min": wake_min,
    }


def build_dataframe(input_dir: Path) -> pd.DataFrame:
    txt_files = sorted(input_dir.glob("*.txt"))
    rows: list[dict[str, object]] = []

    for txt_file in txt_files:
        events = _parse_events(txt_file)
        if not events:
            continue

        first_line = txt_file.read_text(errors="ignore").splitlines()[0:40]
        start_clock_sec = None
        for line in first_line:
            if line.startswith("Recording Date"):
                continue
        # Use first event clock as the bedtime anchor when explicit recording start time is absent.
        if events:
            start_clock_sec = int(events[0].onset_sec)
        if start_clock_sec is None:
            start_clock_sec = 0

        # Recompute bedtime anchor from the first parsed event clock text.
        raw_lines = txt_file.read_text(errors="ignore").splitlines()
        start_clock_value = 0
        for line in raw_lines:
            match = LINE_PATTERN.match(line.strip())
            if match:
                start_clock_value = _clock_to_seconds(match.group("clock"))
                break

        features = _compute_features(events, start_clock_value)
        row: dict[str, object] = {
            "record_id": f"capslpdb_{txt_file.stem}",
            **features,
        }
        row.update(
            _derive_qualitative(
                total_sleep_hours=float(features["total_sleep_hours"]),
                sleep_efficiency_pct=float(features["sleep_efficiency_pct"]),
                interruptions=float(features["interruptions"]),
                sleep_onset_latency_min=float(
                    features["sleep_onset_latency_min"]),
                bedtime_min=float(features["bedtime_clock_min"]),
            )
        )
        rows.append(row)

    if not rows:
        raise ValueError("No valid CAP text annotation files were parsed.")

    frame = pd.DataFrame(rows)
    ordered_columns = [
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
    frame = frame[ordered_columns].copy()
    frame = frame.dropna().reset_index(drop=True)
    return frame


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    frame = build_dataframe(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    print("CAP preprocessing complete.")
    print(f"Output rows: {len(frame)}")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
