from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SLEEP_STAGES = {"1", "2", "3", "4", "R"}
WAKE_STAGES = {"W"}
IN_BED_STAGES = SLEEP_STAGES | WAKE_STAGES

MIMIC_REMOTE_ROOT = "https://physionet.org/files/mimic3wdb-matched/1.0/"
MIMIC_LOCAL_ROOT = ROOT / "physionet.org" / \
    "files" / "mimic3wdb-matched" / "1.0"
COMMON_NON_ECG_LABELS = {"ABP", "PAP", "PLETH",
                         "RESP", "CO2", "CVP", "ART", "ICP"}
ECG_LABEL_PATTERN = re.compile(
    r"^(ECG|[IVX]+|AVR|AVL|AVF|V\d+|MCL\d+)$", re.IGNORECASE)


@dataclass
class StageEvent:
    onset_sec: float
    duration_sec: float
    stage: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract benchmark-ready features from local Sleep-EDF data and mirrored/cached MIMIC-III matched numerics records."
    )
    parser.add_argument("--sleep-edf-input", default=str(ROOT /
                        "data" / "raw" / "fast_s3" / "sleep-edfx"))
    parser.add_argument("--sleep-edf-output", default=str(ROOT /
                        "data" / "sleep_edfx_benchmark_features.csv"))
    parser.add_argument("--mimic-input", default=str(MIMIC_LOCAL_ROOT))
    parser.add_argument("--mimic-output", default=str(ROOT /
                        "data" / "mimic3wdb_matched_benchmark_features.csv"))
    parser.add_argument("--max-mimic-records", type=int, default=None)
    parser.add_argument("--download-mimic-missing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


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


def _compute_sleep_edf_benchmark_rows(input_dir: Path) -> pd.DataFrame:
    hypnograms = sorted(input_dir.glob("**/*-Hypnogram.edf"))
    rows: list[dict[str, object]] = []

    for hypnogram in hypnograms:
        record_base = hypnogram.stem.replace("-Hypnogram", "")
        meta, labels, samples_per_record, payload = _read_edf_header(hypnogram)
        if "EDF Annotations" not in labels or len(samples_per_record) != 1:
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

        stage_events = sorted(stage_events, key=lambda event: event.onset_sec)
        record_start = min(event.onset_sec for event in stage_events)
        in_bed_duration = sum(
            event.duration_sec for event in stage_events if event.stage in IN_BED_STAGES)
        sleep_duration = sum(
            event.duration_sec for event in stage_events if event.stage in SLEEP_STAGES)
        first_sleep = next(
            (event.onset_sec for event in stage_events if event.stage in SLEEP_STAGES), record_start)
        last_sleep_end = max(
            (event.onset_sec + event.duration_sec for event in stage_events if event.stage in SLEEP_STAGES),
            default=record_start,
        )

        interruption_count = 0
        wake_after_sleep_onset = 0.0
        transition_count = 0
        in_wake_bout = False
        previous_stage: str | None = None
        stage_totals = {stage: 0.0 for stage in IN_BED_STAGES}

        for event in stage_events:
            if event.stage in stage_totals:
                stage_totals[event.stage] += event.duration_sec
            if previous_stage is not None and previous_stage != event.stage:
                transition_count += 1
            previous_stage = event.stage
            if event.onset_sec < first_sleep or event.onset_sec >= last_sleep_end:
                continue
            if event.stage in WAKE_STAGES:
                wake_after_sleep_onset += event.duration_sec
                if event.duration_sec >= 30 and not in_wake_bout:
                    interruption_count += 1
                    in_wake_bout = True
            elif event.stage in SLEEP_STAGES:
                in_wake_bout = False

        start_minutes = _parse_start_minutes(meta.get("start_time", ""))
        bedtime_clock_min = (
            start_minutes + record_start / 60.0) % (24.0 * 60.0)
        total_sleep_hours = sleep_duration / 3600.0
        sleep_efficiency_pct = (
            sleep_duration / in_bed_duration) * 100.0 if in_bed_duration > 0 else 0.0

        n3_n4_duration = stage_totals.get(
            "3", 0.0) + stage_totals.get("4", 0.0)
        rows.append(
            {
                "record_id": f"sleepedfx_{record_base}",
                "total_sleep_hours": total_sleep_hours,
                "sleep_efficiency_pct": sleep_efficiency_pct,
                "interruptions": float(interruption_count),
                "sleep_onset_latency_min": (first_sleep - record_start) / 60.0,
                "wake_after_sleep_onset_min": wake_after_sleep_onset / 60.0,
                "wake_after_sleep_onset_pct": (wake_after_sleep_onset / in_bed_duration) * 100.0 if in_bed_duration > 0 else 0.0,
                "rem_pct_of_sleep": (stage_totals.get("R", 0.0) / sleep_duration) * 100.0 if sleep_duration > 0 else 0.0,
                "n1_pct_of_sleep": (stage_totals.get("1", 0.0) / sleep_duration) * 100.0 if sleep_duration > 0 else 0.0,
                "n2_pct_of_sleep": (stage_totals.get("2", 0.0) / sleep_duration) * 100.0 if sleep_duration > 0 else 0.0,
                "n3_n4_pct_of_sleep": (n3_n4_duration / sleep_duration) * 100.0 if sleep_duration > 0 else 0.0,
                "stage_transition_count": float(transition_count),
                "bedtime_clock_min": bedtime_clock_min,
            }
        )

    if not rows:
        raise ValueError(
            "No valid Sleep-EDF hypnogram benchmark rows were parsed.")

    frame = pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)
    return frame


def _ensure_remote_file(local_root: Path, relative_path: str) -> Path | None:
    local_path = local_root / relative_path
    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = MIMIC_REMOTE_ROOT + relative_path
    try:
        with urlopen(url, timeout=60) as response, local_path.open("wb") as handle:
            handle.write(response.read())
    except (HTTPError, URLError, TimeoutError):
        return None
    return local_path


def _parse_mimic_header(header_path: Path) -> tuple[int, float, int, list[dict[str, object]]]:
    lines = [line.strip()
             for line in header_path.read_text().splitlines() if line.strip()]
    header_tokens = lines[0].split()
    signal_count = int(header_tokens[1])
    sample_rate_token = header_tokens[2]
    if "/" in sample_rate_token:
        sample_rate = float(sample_rate_token.split("/", 1)[0])
    else:
        sample_rate = float(sample_rate_token)
    sample_count = int(header_tokens[3])

    signal_rows = []
    for line in lines[1: 1 + signal_count]:
        tokens = line.split()
        gain_token = tokens[2]
        try:
            gain = float(gain_token.split("/", 1)[0])
        except ValueError:
            gain = 1.0
        units = gain_token.split("/", 1)[1] if "/" in gain_token else ""
        signal_rows.append(
            {
                "dat_filename": tokens[0],
                "gain": gain if gain != 0.0 else 1.0,
                "units": units,
                "label": tokens[-1],
            }
        )

    return signal_count, sample_rate, sample_count, signal_rows


def _load_mimic_numeric_matrix(dat_path: Path, signal_count: int, sample_count: int) -> np.ndarray:
    raw = np.fromfile(dat_path, dtype="<i2")
    usable = (raw.size // signal_count) * signal_count
    raw = raw[:usable]
    matrix = raw.reshape(-1, signal_count)
    if sample_count > 0 and matrix.shape[0] > sample_count:
        matrix = matrix[:sample_count, :]
    return matrix.astype(float)


def _summarize_series(values: np.ndarray, prefix: str) -> dict[str, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(clean)),
        f"{prefix}_std": float(np.std(clean)),
        f"{prefix}_min": float(np.min(clean)),
        f"{prefix}_max": float(np.max(clean)),
    }


def _parse_waveform_ecg_metadata(local_root: Path, waveform_record: str) -> tuple[int, str]:
    header_path = _ensure_remote_file(local_root, f"{waveform_record}.hea")
    if header_path is None:
        return 0, ""

    lines = [line.strip()
             for line in header_path.read_text().splitlines() if line.strip()]
    if not lines:
        return 0, ""

    channel_labels: list[str] = []
    first_line = lines[0].split()
    record_name = first_line[0]
    if "/" in record_name:
        layout_segment = None
        for line in lines[1:]:
            tokens = line.split()
            if len(tokens) >= 1 and tokens[0] != "~":
                layout_segment = tokens[0]
                break
        if layout_segment is None:
            return 0, ""
        layout_path = _ensure_remote_file(local_root, str(
            Path(waveform_record).parent / f"{layout_segment}.hea"))
        if layout_path is None:
            return 0, ""
        layout_lines = [
            line.strip() for line in layout_path.read_text().splitlines() if line.strip()]
        for line in layout_lines[1:]:
            tokens = line.split()
            if tokens:
                channel_labels.append(tokens[-1])
    else:
        signal_count = int(first_line[1])
        for line in lines[1: 1 + signal_count]:
            tokens = line.split()
            if tokens:
                channel_labels.append(tokens[-1])

    ecg_labels = [
        label for label in channel_labels
        if ECG_LABEL_PATTERN.match(label) and label.upper() not in COMMON_NON_ECG_LABELS
    ]
    return len(ecg_labels), ";".join(ecg_labels)


def _extract_mimic_benchmark_rows(
    local_root: Path,
    max_records: int | None,
    download_missing: bool,
    progress_every: int,
) -> pd.DataFrame:
    records_path = local_root / "RECORDS-numerics"
    if not records_path.exists():
        raise ValueError("MIMIC RECORDS-numerics list is missing.")

    waveform_records_path = local_root / "RECORDS-waveforms"
    waveform_records = set()
    if waveform_records_path.exists():
        waveform_records = {
            line.strip()
            for line in waveform_records_path.read_text().splitlines()
            if line.strip()
        }

    rows: list[dict[str, object]] = []
    for index, record in enumerate(records_path.read_text().splitlines(), start=1):
        record = record.strip()
        if not record:
            continue
        if max_records is not None and len(rows) >= max_records:
            break

        header_path = local_root / f"{record}.hea"
        if not header_path.exists() and download_missing:
            header_path = _ensure_remote_file(local_root, f"{record}.hea")
        if header_path is None or not header_path.exists():
            continue

        signal_count, sample_rate, sample_count, signal_rows = _parse_mimic_header(
            header_path)
        if not signal_rows:
            continue

        dat_filename = str(signal_rows[0]["dat_filename"])
        dat_relative = str(Path(record).parent / dat_filename)
        dat_path = local_root / dat_relative
        if not dat_path.exists() and download_missing:
            dat_path = _ensure_remote_file(local_root, dat_relative)
        if dat_path is None or not dat_path.exists():
            continue

        try:
            matrix = _load_mimic_numeric_matrix(
                dat_path, signal_count, sample_count)
        except ValueError:
            continue
        if matrix.size == 0:
            continue

        row: dict[str, object] = {
            "record_id": record,
            "patient_id": Path(record).parts[1] if len(Path(record).parts) > 1 else "",
            "duration_hours": float(matrix.shape[0] / sample_rate / 3600.0) if sample_rate > 0 else float("nan"),
            "sample_rate_hz": sample_rate,
            "sample_count": int(matrix.shape[0]),
        }

        labels_to_prefixes = {
            "HR": "heart_rate",
            "RESP": "resp_rate",
            "SpO2": "spo2",
            "PULSE": "pulse_rate",
        }
        available_labels: list[str] = []
        for column_index, signal_row in enumerate(signal_rows):
            label = str(signal_row["label"])
            if label not in labels_to_prefixes:
                continue
            available_labels.append(label)
            values = matrix[:, column_index]
            values[values <= -32000] = np.nan
            gain = float(signal_row["gain"])
            physical = values / gain if gain not in (0.0, np.nan) else values
            row.update(_summarize_series(physical, labels_to_prefixes[label]))

        waveform_record = record[:-1] if record.endswith("n") else record
        ecg_lead_count = 0
        ecg_lead_labels = ""
        if waveform_record in waveform_records:
            ecg_lead_count, ecg_lead_labels = _parse_waveform_ecg_metadata(
                local_root, waveform_record)

        row["available_numeric_labels"] = ";".join(sorted(available_labels))
        row["waveform_record_id"] = waveform_record if waveform_record in waveform_records else ""
        row["ecg_lead_count"] = ecg_lead_count
        row["ecg_lead_labels"] = ecg_lead_labels
        rows.append(row)

        if progress_every > 0 and len(rows) % progress_every == 0:
            print(
                f"Processed {len(rows)} MIMIC numerics records (scanned {index}).")

    if not rows:
        raise ValueError(
            "No MIMIC benchmark rows could be extracted from available numerics data.")

    return pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()

    sleep_edf_frame = _compute_sleep_edf_benchmark_rows(
        Path(args.sleep_edf_input))
    sleep_edf_output = Path(args.sleep_edf_output)
    sleep_edf_output.parent.mkdir(parents=True, exist_ok=True)
    sleep_edf_frame.to_csv(sleep_edf_output, index=False,
                           quoting=csv.QUOTE_MINIMAL)
    print(
        f"Sleep-EDF benchmark extraction complete: {len(sleep_edf_frame)} rows -> {sleep_edf_output.resolve()}")

    mimic_frame = _extract_mimic_benchmark_rows(
        Path(args.mimic_input),
        max_records=args.max_mimic_records,
        download_missing=args.download_mimic_missing,
        progress_every=args.progress_every,
    )
    mimic_output = Path(args.mimic_output)
    mimic_output.parent.mkdir(parents=True, exist_ok=True)
    mimic_frame.to_csv(mimic_output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(
        f"MIMIC benchmark extraction complete: {len(mimic_frame)} rows -> {mimic_output.resolve()}")


if __name__ == "__main__":
    main()
