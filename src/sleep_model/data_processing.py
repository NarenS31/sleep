from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ProcessedDataset:
    rows: list[dict[str, str]]
    feature_names: list[str]
    weights: np.ndarray
    processed_matrix: np.ndarray
    quality_scores: np.ndarray
    raw_numeric_matrix: np.ndarray
    feature_minimums: np.ndarray
    feature_maximums: np.ndarray
    feature_directions: list[str]
    feature_types: list[str]


def load_schema(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _scaled_feature(value: float, minimum: float, maximum: float, direction: str) -> float:
    if maximum <= minimum:
        raise ValueError("Maximum must be greater than minimum in the schema.")
    normalized = (value - minimum) / (maximum - minimum)
    normalized = float(np.clip(normalized, 0.0, 1.0))
    if direction == "lower_better":
        return 1.0 - normalized
    return normalized


def _encode_quantitative(value: str, config: dict) -> float:
    numeric_value = float(value)
    return numeric_value


def _encode_qualitative(value: str, config: dict) -> float:
    scale = {_normalize_key(label): float(score) for label, score in config["scale"].items()}
    text_value = str(value).strip()

    try:
        ordinal_value = float(text_value)
    except ValueError:
        key = _normalize_key(text_value)
        if key not in scale:
            accepted = ", ".join(sorted(scale))
            raise ValueError(f"Unsupported qualitative response '{value}'. Accepted values: {accepted}")
        ordinal_value = scale[key]

    return ordinal_value


def normalize_feature_matrix(
    raw_numeric_matrix: np.ndarray,
    feature_minimums: np.ndarray,
    feature_maximums: np.ndarray,
    feature_directions: list[str],
) -> np.ndarray:
    normalized_columns: list[np.ndarray] = []
    for index in range(raw_numeric_matrix.shape[1]):
        column = raw_numeric_matrix[:, index]
        normalized_column = [
            _scaled_feature(
                float(value),
                float(feature_minimums[index]),
                float(feature_maximums[index]),
                feature_directions[index],
            )
            for value in column
        ]
        normalized_columns.append(np.array(normalized_column, dtype=float))
    return np.column_stack(normalized_columns)


def load_and_process_dataset(csv_path: str | Path, schema_path: str | Path) -> ProcessedDataset:
    schema = load_schema(schema_path)
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError("The dataset is empty.")

    quantitative = schema["quantitative_features"]
    qualitative = schema["qualitative_features"]
    feature_names = list(quantitative.keys()) + list(qualitative.keys())
    feature_types = ["quantitative"] * len(quantitative) + ["qualitative"] * len(qualitative)
    feature_minimums = np.array(
        [quantitative[name]["min"] for name in quantitative]
        + [min(qualitative[name]["scale"].values()) for name in qualitative],
        dtype=float,
    )
    feature_maximums = np.array(
        [quantitative[name]["max"] for name in quantitative]
        + [max(qualitative[name]["scale"].values()) for name in qualitative],
        dtype=float,
    )
    feature_directions = [quantitative[name]["direction"] for name in quantitative] + [
        qualitative[name]["direction"] for name in qualitative
    ]

    missing_columns = [column for column in feature_names if column not in rows[0]]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    raw_numeric_rows: list[list[float]] = []
    for row in rows:
        encoded_row: list[float] = []
        for feature_name, config in quantitative.items():
            encoded_row.append(_encode_quantitative(row[feature_name], config))
        for feature_name, config in qualitative.items():
            encoded_row.append(_encode_qualitative(row[feature_name], config))
        raw_numeric_rows.append(encoded_row)

    weights = np.array(
        [quantitative[name]["weight"] for name in quantitative]
        + [qualitative[name]["weight"] for name in qualitative],
        dtype=float,
    )
    weights = weights / weights.sum()

    raw_numeric_matrix = np.array(raw_numeric_rows, dtype=float)
    processed_matrix = normalize_feature_matrix(
        raw_numeric_matrix,
        feature_minimums,
        feature_maximums,
        feature_directions,
    )
    quality_scores = processed_matrix @ weights

    return ProcessedDataset(
        rows=rows,
        feature_names=feature_names,
        weights=weights,
        processed_matrix=processed_matrix,
        quality_scores=quality_scores,
        raw_numeric_matrix=raw_numeric_matrix,
        feature_minimums=feature_minimums,
        feature_maximums=feature_maximums,
        feature_directions=feature_directions,
        feature_types=feature_types,
    )
