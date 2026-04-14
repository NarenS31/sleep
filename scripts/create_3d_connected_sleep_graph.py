from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleep_model.data_processing import load_and_process_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive 3D connected graph for sleep feature relationships."
    )
    parser.add_argument("--data", default=str(ROOT /
                        "data" / "all_datasets_model_input.csv"))
    parser.add_argument("--schema", default=str(ROOT /
                        "config" / "feature_schema.json"))
    parser.add_argument("--output", default=str(ROOT /
                        "outputs" / "sleep_features_3d_connected.html"))
    parser.add_argument("--max-points", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_column(rows: list[dict[str, str]], candidates: list[str], fallback: str) -> str:
    if not rows:
        return fallback
    keys = set(rows[0].keys())
    for candidate in candidates:
        if candidate in keys:
            return candidate
    return fallback


def encode_scale(values: pd.Series, scale: dict[str, float]) -> np.ndarray:
    key_scale = {key.strip().lower(): float(value)
                 for key, value in scale.items()}

    encoded: list[float] = []
    for raw_value in values.astype(str):
        text = raw_value.strip().lower()
        if text in key_scale:
            encoded.append(key_scale[text])
            continue
        try:
            encoded.append(float(raw_value))
        except ValueError:
            encoded.append(np.nan)
    return np.array(encoded, dtype=float)


def display_name(column_name: str) -> str:
    aliases = {
        "sleep_efficiency_pct": "Sleep Efficiency (%)",
        "sleep_onset_latency_min": "Sleep Onset Latency (min)",
        "wake_variability_min": "Wake Variability (min)",
        "bedtime_variability_min": "Bedtime Variability (min)",
        "screen_time_before_bed": "Screen Time Before Bed (hours)",
    }
    if column_name in aliases:
        return aliases[column_name]

    words = column_name.replace("_", " ").split()
    return " ".join(word.capitalize() for word in words)


def build_edges(
    correlation_frame: pd.DataFrame,
    focus_feature: str,
    min_correlation: float,
    top_edges_per_node: int,
    max_edges: int,
) -> list[dict[str, float | str]]:
    columns = list(correlation_frame.columns)
    selected_pairs: set[tuple[str, str]] = set()

    for feature in columns:
        ranked: list[tuple[float, str, str]] = []
        for other in columns:
            if other == feature:
                continue
            weight = float(correlation_frame.loc[feature, other])
            if pd.isna(weight):
                continue
            ranked.append((abs(weight), feature, other))
        ranked.sort(reverse=True)
        for absolute_weight, left, right in ranked[:top_edges_per_node]:
            if absolute_weight < min_correlation:
                continue
            selected_pairs.add(tuple(sorted((left, right))))

    for feature in columns:
        if feature != focus_feature:
            selected_pairs.add(tuple(sorted((focus_feature, feature))))

    edges = []
    for left, right in selected_pairs:
        value = float(correlation_frame.loc[left, right])
        edges.append(
            {
                "from": left,
                "to": right,
                "value": value,
                "abs_value": abs(value),
            }
        )

    edges.sort(key=lambda edge: (
        edge["abs_value"], edge["from"], edge["to"]), reverse=True)
    return edges[:max_edges]


def main() -> None:
    args = parse_args()
    dataset = load_and_process_dataset(args.data, args.schema)

    frame = pd.DataFrame(dataset.rows)
    frame["sleep_quality_score"] = dataset.quality_scores

    id_col = select_column(
        dataset.rows,
        candidates=["record_id", "id"],
        fallback="record_id",
    )
    if id_col not in frame.columns:
        frame[id_col] = [
            f"record_{index + 1:04d}" for index in range(len(frame))]

    feature_columns = [
        name for name in dataset.feature_names if name in frame.columns]
    if not feature_columns:
        raise ValueError("No schema feature columns found in dataset rows.")

    numeric_features = pd.DataFrame(
        dataset.raw_numeric_matrix, columns=dataset.feature_names)
    graph_frame = pd.DataFrame({
        id_col: frame[id_col].astype(str),
        "sleep_quality_score": dataset.quality_scores,
    })
    for feature_name in feature_columns:
        graph_frame[feature_name] = pd.to_numeric(
            numeric_features[feature_name], errors="coerce")

    graph_frame = graph_frame.dropna(
        subset=feature_columns + ["sleep_quality_score"]).copy()

    if graph_frame.empty:
        raise ValueError(
            "No valid rows available for 3D graph after numeric conversion.")

    if len(graph_frame) > args.max_points:
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(graph_frame.index.to_numpy(),
                              size=args.max_points, replace=False)
        graph_frame = graph_frame.loc[selected].copy()

    graph_frame = graph_frame.sort_values(
        "sleep_quality_score").reset_index(drop=True)

    metric_mins = graph_frame[feature_columns].min()
    metric_maxs = graph_frame[feature_columns].max()
    denom = (metric_maxs - metric_mins).replace(0, 1.0)
    normalized = (graph_frame[feature_columns] - metric_mins) / denom

    record_ids = graph_frame[id_col].astype(str).tolist()
    raw_values = graph_frame[feature_columns].round(4).values.tolist()
    normalized_values = normalized.round(6).values.tolist()
    scores = graph_frame["sleep_quality_score"].astype(float).tolist()

    metric_meta = []
    for feature_name in feature_columns:
        metric_meta.append(
            {
                "id": feature_name,
                "label": display_name(feature_name),
                "minimum": float(metric_mins[feature_name]),
                "maximum": float(metric_maxs[feature_name]),
                "mean": float(graph_frame[feature_name].mean()),
            }
        )

    correlation_frame = graph_frame[feature_columns +
                                    ["sleep_quality_score"]].corr(method="pearson").fillna(0.0)
    focus_feature = "sleep_quality_score"
    axis_features = {
        "x": "sleep_efficiency_pct" if "sleep_efficiency_pct" in correlation_frame.columns else feature_columns[0],
        "y": "screen_time_before_bed" if "screen_time_before_bed" in correlation_frame.columns else feature_columns[min(1, len(feature_columns) - 1)],
        "z": "perceived_stress" if "perceived_stress" in correlation_frame.columns else feature_columns[min(2, len(feature_columns) - 1)],
    }
    edges = build_edges(
        correlation_frame=correlation_frame,
        focus_feature=focus_feature,
        min_correlation=0.18,
        top_edges_per_node=3,
        max_edges=80,
    )

    node_coords: dict[str, tuple[float, float, float]] = {}
    network_nodes = []
    for feature in correlation_frame.columns:
        corr_focus = 1.0 if feature == focus_feature else float(
            correlation_frame.loc[focus_feature, feature])
        x_coord = float(correlation_frame.loc[feature, axis_features["x"]])
        y_coord = float(correlation_frame.loc[feature, axis_features["y"]])
        z_coord = float(correlation_frame.loc[feature, axis_features["z"]])
        if feature == focus_feature:
            x_coord += 0.08
            y_coord += 0.08
            z_coord += 0.08

        node_coords[feature] = (x_coord, y_coord, z_coord)
        network_nodes.append(
            {
                "id": feature,
                "label": "Sleep Quality Score" if feature == focus_feature else display_name(feature),
                "size": round(10 + 18 * abs(corr_focus), 3),
                "color": "#1679ab" if feature == focus_feature else "#0d9488",
                "hover": (
                    f"{'Sleep Quality Score' if feature == focus_feature else display_name(feature)}<br>"
                    f"Correlation with sleep quality: {corr_focus:.3f}<br>"
                    f"Position ({x_coord:.3f}, {y_coord:.3f}, {z_coord:.3f})"
                ),
                "x": x_coord,
                "y": y_coord,
                "z": z_coord,
            }
        )

    positive_x: list[float | None] = []
    positive_y: list[float | None] = []
    positive_z: list[float | None] = []
    negative_x: list[float | None] = []
    negative_y: list[float | None] = []
    negative_z: list[float | None] = []

    for edge in edges:
        left = str(edge["from"])
        right = str(edge["to"])
        value = float(edge["value"])
        lx, ly, lz = node_coords[left]
        rx, ry, rz = node_coords[right]
        if value >= 0:
            positive_x.extend([lx, rx, None])
            positive_y.extend([ly, ry, None])
            positive_z.extend([lz, rz, None])
        else:
            negative_x.extend([lx, rx, None])
            negative_y.extend([ly, ry, None])
            negative_z.extend([lz, rz, None])

    concept_nodes = [
        {
            "id": "binary_limitations",
            "label": "Binary Limits",
            "stage": "Problem",
            "description": "Binary sleep labels are too simplistic and miss the range of real sleep experiences.",
            "x": 0.0,
            "y": 0.25,
            "z": 0.2,
            "color": "#f97316",
            "size": 18,
            "hover": "Starting point: binary classification is too simplistic for sleep modeling.",
        },
        {
            "id": "multi_dimensional_scoring",
            "label": "Multi-Dimensional Scoring",
            "stage": "Core Shift",
            "description": "A wider scoring system captures multiple sleep traits instead of forcing one yes/no label.",
            "x": 3.2,
            "y": 0.6,
            "z": 0.5,
            "color": "#56b6ff",
            "size": 20,
            "hover": "Primary solution: captures richer sleep behavior than binary labels.",
        },
        {
            "id": "autoencoder",
            "label": "Autoencoder",
            "stage": "Techniques",
            "description": "Learns patterns from data without relying entirely on labels, which makes the model less rigid.",
            "x": 4.0,
            "y": 2.1,
            "z": 0.7,
            "color": "#22c7a5",
            "size": 18,
            "hover": "Learns latent structure without labels, reducing rigid label bias.",
        },
        {
            "id": "latent_representation",
            "label": "Latent Representation",
            "stage": "Techniques",
            "description": "Compresses sleep information into meaningful hidden patterns the model can use.",
            "x": 4.8,
            "y": 3.1,
            "z": 1.1,
            "color": "#22c7a5",
            "size": 15,
            "hover": "Compact learned signal from autoencoder for flexible pattern discovery.",
        },
        {
            "id": "feature_scaling",
            "label": "Feature Scaling",
            "stage": "Techniques",
            "description": "Keeps metrics balanced so one feature does not overpower the rest.",
            "x": 3.8,
            "y": 1.45,
            "z": 1.0,
            "color": "#22c7a5",
            "size": 15,
            "hover": "Standardizes multi-feature inputs for stable training and balanced influence.",
        },
        {
            "id": "bias_reduction",
            "label": "Bias Reduction",
            "stage": "Outcomes",
            "description": "Reducing dependence on simplistic labels lowers model bias.",
            "x": 5.6,
            "y": 2.3,
            "z": 2.1,
            "color": "#f59e0b",
            "size": 17,
            "hover": "Autoencoder + multi-dimensional scoring reduces simplistic label bias.",
        },
        {
            "id": "flexibility",
            "label": "Flexibility",
            "stage": "Outcomes",
            "description": "The model can adapt to different sleep patterns instead of forcing everyone into the same mold.",
            "x": 6.8,
            "y": 2.9,
            "z": 2.8,
            "color": "#f59e0b",
            "size": 17,
            "hover": "Model adapts to diverse sleep patterns and latent structure.",
        },
        {
            "id": "accurate_representation",
            "label": "Accurate Sleep Representation",
            "stage": "Outcomes",
            "description": "The final result is a more realistic picture of sleep quality.",
            "x": 8.2,
            "y": 3.6,
            "z": 3.5,
            "color": "#f59e0b",
            "size": 20,
            "hover": "Outcome: better capture of sleep quality than binary classification.",
        },
    ]

    concept_edges = [
        ("binary_limitations", "multi_dimensional_scoring"),
        ("multi_dimensional_scoring", "autoencoder"),
        ("multi_dimensional_scoring", "feature_scaling"),
        ("autoencoder", "latent_representation"),
        ("autoencoder", "bias_reduction"),
        ("multi_dimensional_scoring", "bias_reduction"),
        ("latent_representation", "flexibility"),
        ("feature_scaling", "flexibility"),
        ("bias_reduction", "flexibility"),
        ("flexibility", "accurate_representation"),
        ("bias_reduction", "accurate_representation"),
    ]

    concept_lookup = {node["id"]: node for node in concept_nodes}
    concept_x: list[float | None] = []
    concept_y: list[float | None] = []
    concept_z: list[float | None] = []
    for left, right in concept_edges:
        concept_x.extend([float(concept_lookup[left]["x"]),
                         float(concept_lookup[right]["x"]), None])
        concept_y.extend([float(concept_lookup[left]["y"]),
                         float(concept_lookup[right]["y"]), None])
        concept_z.extend([float(concept_lookup[left]["z"]),
                         float(concept_lookup[right]["z"]), None])

    payload = {
        "title": "Sleep Modeling Story Map (3D)",
        "subtitle": "Use Method Overview for the model narrative, or pick one metric to inspect individual sleep patterns.",
        "record_ids": record_ids,
        "metric_meta": metric_meta,
        "raw_values": raw_values,
        "normalized_values": normalized_values,
        "scores": scores,
        "record_count": len(record_ids),
        "metric_count": len(feature_columns),
        "network": {
            "positive_x": positive_x,
            "positive_y": positive_y,
            "positive_z": positive_z,
            "negative_x": negative_x,
            "negative_y": negative_y,
            "negative_z": negative_z,
            "nodes_x": [float(node["x"]) for node in network_nodes],
            "nodes_y": [float(node["y"]) for node in network_nodes],
            "nodes_z": [float(node["z"]) for node in network_nodes],
            "node_sizes": [float(node["size"]) for node in network_nodes],
            "node_colors": [str(node["color"]) for node in network_nodes],
            "node_text": [str(node["label"]) for node in network_nodes],
            "node_hover": [str(node["hover"]) for node in network_nodes],
            "axis_labels": {
                "x": display_name(axis_features["x"]),
                "y": display_name(axis_features["y"]),
                "z": display_name(axis_features["z"]),
            },
            "edge_count": len(edges),
        },
        "concept_map": {
            "nodes": [
                {
                    "id": str(node["id"]),
                    "label": str(node["label"]),
                    "stage": str(node["stage"]),
                    "description": str(node["description"]),
                    "color": str(node["color"]),
                }
                for node in concept_nodes
            ],
            "lines_x": concept_x,
            "lines_y": concept_y,
            "lines_z": concept_z,
            "nodes_x": [float(node["x"]) for node in concept_nodes],
            "nodes_y": [float(node["y"]) for node in concept_nodes],
            "nodes_z": [float(node["z"]) for node in concept_nodes],
            "node_text": [str(node["label"]) for node in concept_nodes],
            "node_hover": [str(node["hover"]) for node in concept_nodes],
            "node_colors": [str(node["color"]) for node in concept_nodes],
            "node_sizes": [float(node["size"]) for node in concept_nodes],
            "axis_labels": {
                "x": "Model Type",
                "y": "Techniques",
                "z": "Outcomes",
            },
            "range": {
                "x": [-0.5, 8.8],
                "y": [0.0, 4.0],
                "z": [0.0, 3.9],
            },
        },
    }

    html_template = """<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Sleep Modeling Story Map (3D)</title>
    <script src="./plotly-2.35.2.min.js"></script>
    <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
    <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
    <link href=\"https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap\" rel=\"stylesheet\">
    <style>
        :root {
            --ink: #e8f1fb;
            --muted: #9eb2ca;
            --panel: rgba(10, 23, 44, 0.78);
            --line: rgba(120, 173, 224, 0.18);
            --accent: #56b6ff;
            --good: #22c7a5;
        }
        * { box-sizing: border-box; }
        html, body { margin: 0; min-height: 100%; color: var(--ink); font-family: 'Space Grotesk', sans-serif; }
        body {
            background:
                radial-gradient(1100px 700px at 12% -10%, rgba(63, 150, 255, 0.34), transparent 55%),
                radial-gradient(900px 540px at 95% 0%, rgba(34, 199, 165, 0.22), transparent 58%),
                linear-gradient(180deg, #08111f 0%, #0d1a2e 48%, #07101d 100%);
        }
        .page { max-width: 1600px; margin: 0 auto; padding: 20px; }
        .topbar { margin-bottom: 14px; }
        h1 { margin: 0; font-size: clamp(22px, 2.4vw, 34px); letter-spacing: -0.02em; }
        .subtitle { margin: 8px 0 0 0; color: var(--muted); font-size: 15px; max-width: 900px; }
        .layout { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 16px; align-items: stretch; }
        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            backdrop-filter: blur(8px);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.34);
        }
        .main-panel { padding: 16px; }
        #plot {
            width: 100%;
            height: 80vh;
            min-height: 640px;
            border-radius: 16px;
            background:
                radial-gradient(circle at top, rgba(86, 182, 255, 0.1), transparent 35%),
                linear-gradient(180deg, rgba(10, 22, 40, 0.94), rgba(6, 15, 28, 0.98));
        }
        .hidden { display: none; }
        .overview { display: grid; gap: 18px; }
        .overview-hero {
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(240px, 0.8fr);
            gap: 14px;
            align-items: stretch;
        }
        .overview-banner,
        .overview-summary {
            border: 1px solid var(--line);
            border-radius: 18px;
            background:
                linear-gradient(180deg, rgba(16, 34, 63, 0.94), rgba(10, 21, 39, 0.94));
            padding: 18px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }
        .overview-kicker {
            font-size: 11px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #8fc8ff;
            margin-bottom: 8px;
        }
        .overview-banner h2,
        .overview-summary h3 {
            margin: 0;
            letter-spacing: -0.03em;
            color: #ffffff;
        }
        .overview-banner h2 {
            font-size: clamp(24px, 2.2vw, 36px);
            line-height: 1.05;
            max-width: 11ch;
        }
        .overview-banner p,
        .overview-summary p {
            margin: 10px 0 0 0;
            color: #c8d8eb;
            font-size: 14px;
            line-height: 1.5;
        }
        .overview-summary-metric {
            margin-top: 14px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 12px;
            border-radius: 999px;
            background: rgba(86, 182, 255, 0.08);
            border: 1px solid rgba(86, 182, 255, 0.16);
            color: #dff0ff;
            font-size: 13px;
        }
        .overview-flow {
            display: grid;
            grid-template-columns: minmax(180px, 1fr) 48px minmax(200px, 1.1fr) 48px minmax(220px, 1.2fr) 48px minmax(220px, 1.2fr);
            gap: 14px;
            align-items: start;
            min-height: 360px;
        }
        .overview-column {
            display: grid;
            gap: 12px;
            padding: 16px;
            border: 1px solid rgba(120, 173, 224, 0.1);
            border-radius: 18px;
            background: rgba(9, 19, 35, 0.48);
        }
        .stage-title {
            font-size: 11px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #86a4c5;
        }
        .overview-node {
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 14px 14px 16px 14px;
            background: linear-gradient(180deg, rgba(16, 35, 64, 0.95), rgba(10, 22, 41, 0.95));
            color: #f3f7fd;
            font-weight: 700;
            line-height: 1.3;
            box-shadow: 0 12px 24px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.03);
            position: relative;
        }
        .overview-node small {
            display: block;
            margin-top: 8px;
            font-weight: 500;
            color: #b8cae0;
            font-size: 12px;
            line-height: 1.45;
        }
        .overview-node.problem { border-color: rgba(249, 115, 22, 0.42); }
        .overview-node.shift { border-color: rgba(86, 182, 255, 0.42); }
        .overview-node.technique { border-color: rgba(34, 199, 165, 0.38); }
        .overview-node.outcome { border-color: rgba(245, 158, 11, 0.38); }
        .overview-arrow {
            align-self: center;
            justify-self: center;
            color: #8fc8ff;
            font-size: 22px;
            font-weight: 700;
            width: 38px;
            height: 38px;
            border-radius: 999px;
            display: grid;
            place-items: center;
            border: 1px solid rgba(143, 200, 255, 0.18);
            background: rgba(13, 29, 53, 0.84);
        }
        .overview-links {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .overview-link-chip {
            padding: 8px 12px;
            border-radius: 999px;
            background: linear-gradient(180deg, rgba(13, 29, 53, 0.96), rgba(10, 22, 41, 0.96));
            border: 1px solid var(--line);
            color: #d8e6f7;
            font-size: 12px;
        }
        .overview-descriptions {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-top: 4px;
        }
        .description-card {
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 14px;
            background: linear-gradient(180deg, rgba(11, 25, 46, 0.94), rgba(9, 20, 36, 0.94));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .description-card h4 {
            margin: 0 0 8px 0;
            font-size: 14px;
            color: #ffffff;
        }
        .description-card p {
            margin: 0;
            font-size: 13px;
            line-height: 1.5;
            color: #c9d8ea;
        }
        .sidebar { padding: 16px; background: linear-gradient(180deg, rgba(11, 25, 46, 0.92), rgba(8, 17, 31, 0.96)); }
        .label { margin: 0 0 8px 0; font-weight: 700; font-size: 12px; letter-spacing: 0.06em; color: #b9cce3; text-transform: uppercase; }
        select {
            width: 100%;
            border: 1px solid var(--line);
            border-radius: 10px;
            background: rgba(16, 31, 56, 0.96);
            color: var(--ink);
            padding: 10px;
            font-family: 'Space Grotesk', sans-serif;
            margin-bottom: 14px;
        }
        .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }
        .card { border: 1px solid var(--line); border-radius: 12px; padding: 10px; background: rgba(13, 29, 53, 0.92); }
        .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
        .v { font-weight: 700; font-size: 18px; margin-top: 3px; }
        .person { border: 1px solid var(--line); border-radius: 12px; padding: 12px; background: rgba(13, 29, 53, 0.92); margin-top: 8px; }
        .person h3 { margin: 0 0 8px 0; font-size: 15px; }
        .person-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 12px; color: #d2dff0; }
        .mono { font-family: 'IBM Plex Mono', monospace; }
        .hint { margin-top: 12px; font-size: 12px; color: var(--muted); line-height: 1.4; }
        .guide { margin-top: 12px; font-size: 12px; color: #d8e6f7; line-height: 1.5; }
        .guide strong { color: #ffffff; display: block; margin-bottom: 6px; }
        @media (max-width: 1080px) {
            .layout { grid-template-columns: 1fr; }
            #plot { height: 66vh; min-height: 520px; }
            .overview-hero { grid-template-columns: 1fr; }
            .overview-flow { grid-template-columns: 1fr; }
            .overview-arrow { transform: rotate(90deg); }
            .overview-descriptions { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class=\"page\">
        <div class=\"topbar\">
            <h1 id=\"title\">Sleep Modeling Story Map (3D)</h1>
            <p class=\"subtitle\" id=\"subtitle\">Use Method Overview for the model narrative, or pick one metric to inspect individual sleep patterns.</p>
        </div>
        <div class=\"layout\">
            <section class=\"panel main-panel\">
                <div id=\"overview\" class=\"overview hidden\"></div>
                <div id=\"plot\"></div>
            </section>
            <aside class=\"panel sidebar\">
                <p class=\"label\">Metric Focus</p>
                <select id=\"metric-select\"></select>

                <div id="person-focus-group">
                    <p class="label">Highlight Person</p>
                    <select id="person-select"></select>
                </div>

                <div class="stats" id="stats-block">
                    <div class=\"card\"><div class=\"k\">People</div><div class=\"v\" id=\"count-people\"></div></div>
                    <div class=\"card\"><div class=\"k\">Metrics</div><div class=\"v\" id=\"count-metrics\"></div></div>
                </div>

                <div class=\"person\" id=\"person-card\">
                    <h3 id=\"person-name\"></h3>
                    <div class=\"person-grid\" id=\"person-grid\"></div>
                </div>

                <p class=\"guide\" id=\"how-to-read\">
                    <strong>How to read this</strong>
                    Method Overview: model story from problem to outcomes.<br>
                    Focused Metric: each point is one person for the selected metric.
                </p>

                <p class=\"hint\" id=\"mode-hint\"></p>
            </aside>
        </div>
    </div>
    <script>
        const payload = __PAYLOAD__;

        const metricIds = payload.metric_meta.map((item) => item.id);
        const metricLabels = payload.metric_meta.map((item) => item.label);
        const metricSpacing = 3.2;
        const metricCenters = metricIds.map((_, index) => index * metricSpacing);
        const scorePoints = payload.scores.map((value) => Math.round(value * 100));

        const qualitativeLabelMap = {
            perceived_stress: ['Not at all', 'A little', 'Moderate', 'Very', 'Extreme'],
            morning_mood: ['Awful', 'Poor', 'Okay', 'Good', 'Excellent'],
            daytime_sleepiness: ['Never', 'Rarely', 'Sometimes', 'Often', 'Constant'],
            screen_time_before_bed: ['None', 'Brief', 'Moderate', 'Heavy', 'Very heavy'],
        };

        function formatMetricValue(metricId, rawValue) {
            const numeric = Number(rawValue);
            if (Number.isNaN(numeric)) {
                return String(rawValue);
            }

            if (qualitativeLabelMap[metricId]) {
                const idx = Math.max(0, Math.min(4, Math.round(numeric) - 1));
                return qualitativeLabelMap[metricId][idx];
            }

            if (metricId === 'sleep_efficiency_pct') {
                return Math.round(numeric) + '%';
            }
            if (metricId === 'total_sleep_hours') {
                return numeric.toFixed(1) + ' hours';
            }
            if (metricId === 'interruptions') {
                return Math.round(numeric) + ' interruptions';
            }
            if (metricId.endsWith('_min')) {
                return Math.round(numeric) + ' min';
            }
            return numeric.toFixed(1);
        }

        const title = document.getElementById('title');
        const subtitle = document.getElementById('subtitle');
        const metricSelect = document.getElementById('metric-select');
        const personSelect = document.getElementById('person-select');
        const countPeople = document.getElementById('count-people');
        const countMetrics = document.getElementById('count-metrics');
        const personName = document.getElementById('person-name');
        const personGrid = document.getElementById('person-grid');
        const modeHint = document.getElementById('mode-hint');
        const howToRead = document.getElementById('how-to-read');
        const overview = document.getElementById('overview');
        const plot = document.getElementById('plot');
        const personFocusGroup = document.getElementById('person-focus-group');
        const statsBlock = document.getElementById('stats-block');
        const personCard = document.getElementById('person-card');

        title.textContent = payload.title;
        subtitle.textContent = payload.subtitle;
        countPeople.textContent = payload.record_count;
        countMetrics.textContent = payload.metric_count;

        metricSelect.innerHTML = '';
        const allOption = document.createElement('option');
        allOption.value = '__all__';
        allOption.textContent = 'Method Overview (Concept Map)';
        metricSelect.appendChild(allOption);

        payload.metric_meta.forEach((metric) => {
            const option = document.createElement('option');
            option.value = metric.id;
            option.textContent = metric.label;
            metricSelect.appendChild(option);
        });

        personSelect.innerHTML = '';
        const cohortOption = document.createElement('option');
        cohortOption.value = '__none__';
        cohortOption.textContent = 'No Person Highlight';
        personSelect.appendChild(cohortOption);
        payload.record_ids.forEach((recordId, index) => {
            const option = document.createElement('option');
            option.value = String(index);
            option.textContent = recordId;
            personSelect.appendChild(option);
        });

        function percentile(sortedValues, p) {
            if (!sortedValues.length) return 0;
            const position = Math.min(sortedValues.length - 1, Math.max(0, Math.round((sortedValues.length - 1) * p)));
            return sortedValues[position];
        }

        function renderMethodOverview() {
            const stageOrder = ['Problem', 'Core Shift', 'Techniques', 'Outcomes'];
            const stageClassMap = {
                Problem: 'problem',
                'Core Shift': 'shift',
                Techniques: 'technique',
                Outcomes: 'outcome',
            };
            const stageNodes = Object.fromEntries(stageOrder.map((stage) => [stage, []]));
            payload.concept_map.nodes.forEach((node) => {
                stageNodes[node.stage].push(node);
            });

            const columnHtml = stageOrder.map((stage) => {
                const cards = stageNodes[stage].map((node) => (
                    '<div class="overview-node ' + stageClassMap[stage] + '">' + node.label + '<small>' + node.description + '</small></div>'
                )).join('');
                return '<div class="overview-column"><div class="stage-title">' + stage + '</div>' + cards + '</div>';
            });

            const flowHtml =
                columnHtml[0] +
                '<div class="overview-arrow">-&gt;</div>' +
                columnHtml[1] +
                '<div class="overview-arrow">-&gt;</div>' +
                columnHtml[2] +
                '<div class="overview-arrow">-&gt;</div>' +
                columnHtml[3];

            const linkChips = [
                'Binary Limits -> Multi-Dimensional Scoring',
                'Scoring -> Autoencoder',
                'Scoring -> Feature Scaling',
                'Autoencoder -> Latent Representation',
                'Bias Reduction -> Flexibility -> Accurate Sleep Representation',
            ].map((text) => '<span class="overview-link-chip">' + text + '</span>').join('');

            const descriptions = payload.concept_map.nodes.map((node) => (
                '<div class="description-card">' +
                '<h4>' + node.label + '</h4>' +
                '<p>' + node.description + '</p>' +
                '</div>'
            )).join('');

            overview.innerHTML =
                '<div class="overview-hero">' +
                '<div class="overview-banner">' +
                '<div class="overview-kicker">Method Overview</div>' +
                '<h2>From binary limits to a better model of sleep</h2>' +
                '<p>This slide shows why the project moves beyond simple classification and how the modeling choices improve sleep representation.</p>' +
                '</div>' +
                '<div class="overview-summary">' +
                '<div class="overview-kicker">Takeaway</div>' +
                '<h3>Better sleep modeling needs richer structure</h3>' +
                '<p>Multi-dimensional scoring and an autoencoder work together to reduce bias, increase flexibility, and represent sleep more accurately.</p>' +
                '<div class="overview-summary-metric">8 key method nodes connected in one flow</div>' +
                '</div>' +
                '</div>' +
                '<div class="overview-flow">' + flowHtml + '</div>' +
                '<div class="overview-links">' + linkChips + '</div>' +
                '<div class="overview-descriptions">' + descriptions + '</div>';
        }

        function buildFeatureColumn(metricIndex) {
            const x = [];
            const y = [];
            const z = [];
            const hover = [];
            const sizes = [];
            const goldenAngle = Math.PI * (3 - Math.sqrt(5));
            const metricCenter = metricCenters[metricIndex];

            for (let i = 0; i < payload.record_ids.length; i += 1) {
                const recordId = payload.record_ids[i];
                const raw = payload.raw_values[i][metricIndex];
                const norm = payload.normalized_values[i][metricIndex];
                const score = payload.scores[i];
                const scorePoint = scorePoints[i];

                // Non-repeating spiral spread to avoid ring overlap at larger cohort sizes.
                const angle = i * goldenAngle;
                const radial = 0.018 * Math.sqrt(i + 1);

                const xOffset = Math.cos(angle) * radial * 3.7;
                const yOffset = Math.sin(angle) * radial * 0.58;
                const zOffset = (Math.cos(angle * 0.7) * radial * 7.0) + (Math.sin(angle * 1.3) * 1.2);

                x.push(metricCenter + xOffset);
                y.push(norm + yOffset);
                z.push(scorePoint + zOffset);
                hover.push(
                    'Person: ' + recordId + '<br>' +
                    metricLabels[metricIndex] + ': ' + formatMetricValue(metricIds[metricIndex], raw) + '<br>' +
                    'Sleep score: ' + scorePoint + ' / 100'
                );
                sizes.push(3.8 + (score * 2.2));
            }
            return { x, y, z, hover, sizes };
        }

        function buildHighlightTrace(personIndex, selectedMetricId) {
            if (personIndex === null) {
                return null;
            }

            if (selectedMetricId === '__all__') {
                return null;
            }

            const score = payload.scores[personIndex];
            const scorePoint = Math.round(score * 100);
            const x = [];
            const y = [];
            const z = [];
            const hover = [];
            const recordId = payload.record_ids[personIndex];

            const selectedMetricIndex = metricIds.indexOf(selectedMetricId);
            x.push(metricCenters[selectedMetricIndex]);
            y.push(payload.normalized_values[personIndex][selectedMetricIndex]);
            z.push(scorePoint);
            hover.push(
                'Person: ' + recordId + '<br>' +
                metricLabels[selectedMetricIndex] + ': ' + formatMetricValue(metricIds[selectedMetricIndex], payload.raw_values[personIndex][selectedMetricIndex]) + '<br>' +
                'Sleep score: ' + scorePoint + ' / 100'
            );

            return { x, y, z, hover, recordId };
        }

        function renderPersonCard(personIndex, selectedMetricId) {
            if (personIndex === null) {
                personName.textContent = 'Cohort Summary';
                personGrid.innerHTML = '';

                const sortedScores = [...payload.scores].sort((a, b) => a - b);
                const rows = [
                    ['Median sleep score', Math.round(percentile(sortedScores, 0.5) * 100) + ' / 100'],
                    ['Lower quartile', Math.round(percentile(sortedScores, 0.25) * 100) + ' / 100'],
                    ['Upper quartile', Math.round(percentile(sortedScores, 0.75) * 100) + ' / 100'],
                ];

                if (selectedMetricId !== '__all__') {
                    const metricIndex = metricIds.indexOf(selectedMetricId);
                    const values = payload.raw_values.map((row) => row[metricIndex]).sort((a, b) => a - b);
                    rows.push([metricLabels[metricIndex] + ' median', formatMetricValue(metricIds[metricIndex], percentile(values, 0.5))]);
                }

                rows.forEach(([k, v]) => {
                    const key = document.createElement('div');
                    key.textContent = k;
                    const value = document.createElement('div');
                    value.className = 'mono';
                    value.textContent = v;
                    personGrid.appendChild(key);
                    personGrid.appendChild(value);
                });
                return;
            }

            personName.textContent = 'Person ' + payload.record_ids[personIndex];
            personGrid.innerHTML = '';
            const score = payload.scores[personIndex];
            const scorePoint = Math.round(score * 100);

            let pairs = [];
            if (selectedMetricId === '__all__') {
                pairs = metricIds.map((id, index) => [metricLabels[index], payload.raw_values[personIndex][index], id]);
            } else {
                const metricIndex = metricIds.indexOf(selectedMetricId);
                pairs = [[metricLabels[metricIndex], payload.raw_values[personIndex][metricIndex], metricIds[metricIndex]]];
            }
            pairs.unshift(['Sleep score', scorePoint + ' / 100', null]);

            pairs.forEach(([k, v, metricId]) => {
                const key = document.createElement('div');
                key.textContent = k;
                const value = document.createElement('div');
                value.className = 'mono';
                if (typeof v === 'string') {
                    value.textContent = v;
                } else {
                    value.textContent = formatMetricValue(metricId, v);
                }
                personGrid.appendChild(key);
                personGrid.appendChild(value);
            });
        }

        function draw() {
            const selectedMetric = metricSelect.value || '__all__';
            const personValue = personSelect.value || '__none__';
            const selectedPerson = personValue === '__none__' ? null : Number(personValue);
            const selectedMetricIndex = selectedMetric === '__all__' ? null : metricIds.indexOf(selectedMetric);
            const selectedMetricCenter = selectedMetricIndex === null ? null : metricCenters[selectedMetricIndex];

            const traces = [];
            const axisTickValues = metricCenters;

            if (selectedMetric === '__all__') {
                plot.classList.add('hidden');
                overview.classList.remove('hidden');
                personFocusGroup.classList.add('hidden');
                statsBlock.classList.add('hidden');
                personCard.classList.add('hidden');
                renderMethodOverview();
                if (typeof Plotly !== 'undefined') {
                    Plotly.purge('plot');
                }
                modeHint.textContent = 'Method Overview: read left to right from problem to solution to outcomes.';
                howToRead.innerHTML = '<strong>How to read this</strong>Top section shows the connected story. Bottom cards explain each node in plain language.';
            } else {
                overview.classList.add('hidden');
                plot.classList.remove('hidden');
                personFocusGroup.classList.remove('hidden');
                statsBlock.classList.remove('hidden');
                personCard.classList.remove('hidden');
                const metricIndex = metricIds.indexOf(selectedMetric);
                const column = buildFeatureColumn(metricIndex);
                traces.push({
                    type: 'scatter3d',
                    mode: 'markers',
                    x: column.x,
                    y: column.y,
                    z: column.z,
                    text: column.hover,
                    hoverinfo: 'text',
                    marker: {
                        size: column.sizes,
                        color: scorePoints,
                        colorscale: 'Turbo',
                        opacity: 0.78,
                        line: { color: 'rgba(255,255,255,0.9)', width: 0.8 },
                        colorbar: { title: 'Sleep score (0-100)' },
                    },
                    name: metricLabels[metricIndex],
                });
                traces.push({
                    type: 'scatter3d',
                    mode: 'markers',
                    x: [metricCenters[metricIndex]],
                    y: [0.5],
                    z: [scorePoints.reduce((sum, value) => sum + value, 0) / scorePoints.length],
                    hovertext: ['Metric node: ' + metricLabels[metricIndex]],
                    hoverinfo: 'text',
                    marker: {
                        size: 18,
                        color: '#1679ab',
                        line: { color: '#ffffff', width: 1.5 },
                    },
                    name: 'Selected metric node',
                });
                modeHint.textContent = metricLabels[metricIndex] + ' view: each point is one person.';
                howToRead.innerHTML = '<strong>How to read this</strong>Higher point = higher sleep score. Hover a point to see the person details.';
                const layout = {
                    margin: { l: 0, r: 0, t: 0, b: 0 },
                    showlegend: true,
                    legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(8,18,33,0.72)', font: { color: '#dce9f8' } },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    scene: {
                        bgcolor: 'rgba(0,0,0,0)',
                        xaxis: {
                            title: 'Selected Metric',
                            tickvals: [selectedMetricCenter],
                            ticktext: [metricLabels[selectedMetricIndex]],
                            tickangle: -25,
                            gridcolor: 'rgba(130, 170, 220, 0.12)',
                            zerolinecolor: 'rgba(130, 170, 220, 0.24)',
                            color: '#dce9f8',
                            range: [selectedMetricCenter - 10.5, selectedMetricCenter + 10.5],
                        },
                        yaxis: {
                            title: 'Relative Metric Level',
                            tickvals: [-0.25, 0.0, 0.5, 1.0, 1.25],
                            ticktext: ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                            range: [-0.55, 1.55],
                            gridcolor: 'rgba(130, 170, 220, 0.12)',
                            zerolinecolor: 'rgba(130, 170, 220, 0.24)',
                            color: '#dce9f8',
                        },
                        zaxis: {
                            title: 'Sleep Score (0-100)',
                            range: [Math.min(...scorePoints) - 18, Math.max(...scorePoints) + 18],
                            gridcolor: 'rgba(130, 170, 220, 0.12)',
                            zerolinecolor: 'rgba(130, 170, 220, 0.24)',
                            color: '#dce9f8',
                        },
                        camera: { eye: { x: 2.1, y: 1.55, z: 1.25 } },
                    },
                };

                Plotly.react('plot', traces, layout, { responsive: true, displaylogo: false });
            }

            const highlight = buildHighlightTrace(selectedPerson, selectedMetric);
            if (selectedMetric !== '__all__' && highlight) {
                traces.push({
                    type: 'scatter3d',
                    mode: 'lines+markers',
                    x: highlight.x,
                    y: highlight.y,
                    z: highlight.z,
                    text: highlight.hover,
                    hoverinfo: 'text',
                    line: { color: 'rgba(233, 94, 62, 0.95)', width: 8 },
                    marker: {
                        size: selectedMetric === '__all__' ? 6 : 8,
                        color: '#f97316',
                        line: { color: '#ffffff', width: 1.2 },
                    },
                    name: 'Highlighted person',
                });
                const layout = {
                    margin: { l: 0, r: 0, t: 0, b: 0 },
                    showlegend: true,
                    legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(8,18,33,0.72)', font: { color: '#dce9f8' } },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    scene: {
                        bgcolor: 'rgba(0,0,0,0)',
                        xaxis: {
                            title: 'Selected Metric',
                            tickvals: [selectedMetricCenter],
                            ticktext: [metricLabels[selectedMetricIndex]],
                            tickangle: -25,
                            gridcolor: 'rgba(130, 170, 220, 0.12)',
                            zerolinecolor: 'rgba(130, 170, 220, 0.24)',
                            color: '#dce9f8',
                            range: [selectedMetricCenter - 10.5, selectedMetricCenter + 10.5],
                        },
                        yaxis: {
                            title: 'Relative Metric Level',
                            tickvals: [-0.25, 0.0, 0.5, 1.0, 1.25],
                            ticktext: ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                            range: [-0.55, 1.55],
                            gridcolor: 'rgba(130, 170, 220, 0.12)',
                            zerolinecolor: 'rgba(130, 170, 220, 0.24)',
                            color: '#dce9f8',
                        },
                        zaxis: {
                            title: 'Sleep Score (0-100)',
                            range: [Math.min(...scorePoints) - 18, Math.max(...scorePoints) + 18],
                            gridcolor: 'rgba(130, 170, 220, 0.12)',
                            zerolinecolor: 'rgba(130, 170, 220, 0.24)',
                            color: '#dce9f8',
                        },
                        camera: { eye: { x: 2.1, y: 1.55, z: 1.25 } },
                    },
                };

                Plotly.react('plot', traces, layout, { responsive: true, displaylogo: false });
            }

            renderPersonCard(selectedPerson, selectedMetric);
        }

        metricSelect.addEventListener('change', draw);
        personSelect.addEventListener('change', draw);
        draw();
    </script>
</body>
</html>
"""

    html = html_template.replace("__PAYLOAD__", json.dumps(payload))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print("3D connected sleep feature graph complete.")
    print(f"Rows plotted: {len(graph_frame)}")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
