from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SLEEP_STAGES = {"1", "2", "3", "4", "R"}
WAKE_STAGES = {"W"}
IN_BED_STAGES = SLEEP_STAGES | WAKE_STAGES


@dataclass
class StageEvent:
    onset_sec: float
    duration_sec: float
    stage: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Sleep-EDF hypnogram EDFs into model input CSV format."
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw/fast_s3/sleep-edfx",
        help="Path to sleep-edfx root directory.",
    )
    parser.add_argument(
        "--output",
        default="data/sleep_edfx_model_input.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def _read_edf_header(path: Path) -> tuple[dict[str, str], list[str], list[int], bytes]:
    with path.open("rb") as handle:
        header = handle.read(256)
        signal_count = int(header[252:256].decode("ascii").strip())
        signal_headers = handle.read(signal_count * 256)
        payload = handle.read()

    labels = []
    cursor = 0
    for _ in range(signal_count):
        labels.append(
            signal_headers[cursor: cursor + 16].decode("latin1").strip())
        cursor += 16

    cursor = signal_count * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    samples_per_record = []
    for _ in range(signal_count):
        samples_per_record.append(
            int(signal_headers[cursor: cursor + 8].decode("ascii").strip()))
        cursor += 8

    meta = {
        "start_date": header[168:176].decode("ascii", errors="ignore").strip(),
        "start_time": header[176:184].decode("ascii", errors="ignore").strip(),
    }
    return meta, labels, samples_per_record, payload


def _tal_events_from_bytes(raw: bytes) -> list[tuple[float, float | None, list[str]]]:
    events: list[tuple[float, float | None, list[str]]] = []
    for tal_blob in raw.split(b"\x00"):
        if not tal_blob:
            continue
        parts = tal_blob.split(b"\x14")
        if not parts:
            continue
        onset_and_duration = parts[0].decode("latin1", errors="ignore").strip()
        if not onset_and_duration:
            continue

        duration_value: float | None = None
        if "\x15" in onset_and_duration:
            onset_text, duration_text = onset_and_duration.split("\x15", 1)
            onset_text = onset_text.strip()
            duration_text = duration_text.strip()
            if duration_text:
                try:
                    duration_value = float(duration_text)
                except ValueError:
                    duration_value = None
        else:
            onset_text = onset_and_duration

        try:
            onset_value = float(onset_text)
        except ValueError:
            continue

        annotations = [
            token.decode("latin1", errors="ignore").strip()
            for token in parts[1:]
            if token.strip()
        ]
        events.append((onset_value, duration_value, annotations))
    return events


def _stage_from_annotation(annotation: str) -> str | None:
    text = annotation.strip()
    if text == "Sleep stage W":
        return "W"
    if text == "Sleep stage R":
        return "R"
    match = re.fullmatch(r"Sleep stage ([1-4])", text)
    if match:
        return match.group(1)
    return None


def _parse_start_minutes(start_time: str) -> float:
    try:
        hh, mm, ss = [int(part) for part in start_time.split(".")]
    except Exception:
        return 0.0
    return hh * 60.0 + mm + (ss / 60.0)


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _bin_label(value: float, edges: list[float], labels: list[str]) -> str:
    bins = [-math.inf] + edges + [math.inf]
    return str(pd.cut(pd.Series([value]), bins=bins, labels=labels, include_lowest=True).iloc[0])


def _derive_qualitative(total_sleep_hours: float, sleep_efficiency_pct: float, interruptions: float, sleep_onset_latency_min: float, bedtime_min: float) -> dict[str, str]:
    stress_score = (8.0 - total_sleep_hours) + \
        (100.0 - sleep_efficiency_pct) / 20.0 + interruptions * 0.2
    mood_score = 5.2 - (8.0 - total_sleep_hours) - \
        (100.0 - sleep_efficiency_pct) / 18.0
    sleepy_score = (8.0 - total_sleep_hours) + sleep_onset_latency_min / 35.0
    screen_proxy = max(0.0, (bedtime_min - 22.5 * 60.0) /
                       35.0) + sleep_onset_latency_min / 45.0

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


def _compute_record_features(events: list[StageEvent], start_minutes: float) -> dict[str, float]:
    if not events:
        raise ValueError("No sleep stage events found.")

    events = sorted(events, key=lambda event: event.onset_sec)
    record_start = min(event.onset_sec for event in events)

    in_bed_dur = sum(
        event.duration_sec for event in events if event.stage in IN_BED_STAGES)
    sleep_dur = sum(
        event.duration_sec for event in events if event.stage in SLEEP_STAGES)

    first_sleep = next(
        (event.onset_sec for event in events if event.stage in SLEEP_STAGES), None)
    if first_sleep is None:
        first_sleep = record_start

    last_sleep_end = max(
        (event.onset_sec +
         event.duration_sec for event in events if event.stage in SLEEP_STAGES),
        default=record_start,
    )

    interruption_count = 0
    in_wake_bout = False
    for event in events:
        if event.onset_sec < first_sleep or event.onset_sec >= last_sleep_end:
            continue
        if event.stage in WAKE_STAGES and event.duration_sec >= 30:
            if not in_wake_bout:
                interruption_count += 1
                in_wake_bout = True
        elif event.stage in SLEEP_STAGES:
            in_wake_bout = False

    total_sleep_hours = _clip(sleep_dur / 3600.0, 3.0, 10.0)
    sleep_efficiency_pct = _clip(
        (sleep_dur / in_bed_dur) * 100.0 if in_bed_dur > 0 else 60.0, 60.0, 100.0)
    sleep_onset_latency_min = _clip(
        (first_sleep - record_start) / 60.0, 0.0, 90.0)
    interruptions = _clip(float(interruption_count), 0.0, 8.0)

    bedtime_min = (start_minutes + (record_start / 60.0)) % (24.0 * 60.0)
    wake_min = (start_minutes + (max(event.onset_sec +
                event.duration_sec for event in events) / 60.0)) % (24.0 * 60.0)

    return {
        "total_sleep_hours": total_sleep_hours,
        "sleep_efficiency_pct": sleep_efficiency_pct,
        "interruptions": interruptions,
        "sleep_onset_latency_min": sleep_onset_latency_min,
        "bedtime_clock_min": bedtime_min,
        "wake_clock_min": wake_min,
    }


def _subject_key(record_id: str) -> str:
    token = record_id.split("-", 1)[0]
    return token[:5]


def build_dataframe(input_dir: Path) -> pd.DataFrame:
    hypnograms = sorted(input_dir.glob("**/*-Hypnogram.edf"))
    rows: list[dict[str, object]] = []

    for hypnogram in hypnograms:
        record_base = hypnogram.stem.replace("-Hypnogram", "")
        meta, labels, samples_per_record, payload = _read_edf_header(hypnogram)
        if "EDF Annotations" not in labels:
            continue
        if len(samples_per_record) != 1:
            continue

        raw_annotation = payload[: samples_per_record[0] * 2]
        tal_events = _tal_events_from_bytes(raw_annotation)
        stage_events: list[StageEvent] = []

        for index, (onset, duration, annotations) in enumerate(tal_events):
            stage = None
            for text in annotations:
                parsed = _stage_from_annotation(text)
                if parsed is not None:
                    stage = parsed
                    break
            if stage is None:
                continue

            if duration is None:
                next_onset = None
                for next_index in range(index + 1, len(tal_events)):
                    if tal_events[next_index][0] > onset:
                        next_onset = tal_events[next_index][0]
                        break
                duration = (
                    next_onset - onset) if next_onset is not None else 30.0
            duration = max(0.0, float(duration))
            if duration == 0.0:
                continue

            stage_events.append(StageEvent(onset_sec=float(
                onset), duration_sec=duration, stage=stage))

        if not stage_events:
            continue

        features = _compute_record_features(
            stage_events, _parse_start_minutes(meta.get("start_time", "")))
        rows.append(
            {
                "record_id": f"sleepedfx_{record_base}",
                "subject_id": _subject_key(record_base),
                **features,
            }
        )

    if not rows:
        raise ValueError("No valid Sleep-EDF hypnogram records were parsed.")

    frame = pd.DataFrame(rows)

    frame["wake_variability_min"] = (
        frame.groupby("subject_id")["wake_clock_min"].transform(
            lambda series: float(series.std(ddof=0) if len(series) > 1 else 0.0))
    )
    frame["bedtime_variability_min"] = (
        frame.groupby("subject_id")["bedtime_clock_min"].transform(
            lambda series: float(series.std(ddof=0) if len(series) > 1 else 0.0))
    )

    frame["wake_variability_min"] = frame["wake_variability_min"].clip(
        0.0, 180.0)
    frame["bedtime_variability_min"] = frame["bedtime_variability_min"].clip(
        0.0, 180.0)

    qualitative = frame.apply(
        lambda row: _derive_qualitative(
            total_sleep_hours=float(row["total_sleep_hours"]),
            sleep_efficiency_pct=float(row["sleep_efficiency_pct"]),
            interruptions=float(row["interruptions"]),
            sleep_onset_latency_min=float(row["sleep_onset_latency_min"]),
            bedtime_min=float(row["bedtime_clock_min"]),
        ),
        axis=1,
        result_type="expand",
    )
    frame = pd.concat([frame, qualitative], axis=1)

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

    print("Sleep-EDF preprocessing complete.")
    print(f"Output rows: {len(frame)}")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
