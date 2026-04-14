from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASE_SLEEP_FEATURES = [
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

LIFESTYLE_FEATURES = [
    "meals_per_day",
    "exercise_days_per_week",
    "screen_hours_before_bed",
    "caffeine_cups_per_day",
    "stress_1_to_10",
    "sunlight_hours_per_day",
    "meal_timing",
    "exercise_time_of_day",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive 3D feature-correlation network from the extended sleep dataset."
    )
    parser.add_argument(
        "--data", default="data/all_datasets_model_input_extended_scaled.csv")
    parser.add_argument(
        "--output", default="outputs/sleep_features_3d_connected_extended.html")
    parser.add_argument("--color", default="sleep_quality_score")
    parser.add_argument("--min-correlation", type=float, default=0.18)
    parser.add_argument("--top-edges-per-node", type=int, default=3)
    parser.add_argument("--max-edges", type=int, default=120)
    parser.add_argument("--x", default="sleep_efficiency_pct")
    parser.add_argument("--y", default="screen_hours_before_bed")
    parser.add_argument("--z", default="stress_1_to_10")
    return parser.parse_args()


def display_name(column_name: str) -> str:
    words = column_name.replace("_pct", " pct").replace("_", " ").split()
    return " ".join(word.upper() if word.lower() == "pct" else word.capitalize() for word in words)


def classify_feature(column_name: str, focus_feature: str) -> str:
    if column_name == focus_feature:
        return "derived"
    if column_name in BASE_SLEEP_FEATURES:
        return "sleep"
    if column_name in LIFESTYLE_FEATURES:
        return "lifestyle"
    return "other"


def build_edges(
    correlations: pd.DataFrame,
    focus_feature: str,
    min_correlation: float,
    top_edges_per_node: int,
    max_edges: int,
) -> list[dict[str, float | str]]:
    columns = list(correlations.columns)
    selected_pairs: set[tuple[str, str]] = set()

    for feature in columns:
        ranked = []
        for other in columns:
            if other == feature:
                continue
            weight = float(correlations.loc[feature, other])
            if pd.isna(weight):
                continue
            ranked.append((abs(weight), feature, other))
        ranked.sort(reverse=True)
        for absolute_weight, left, right in ranked[:top_edges_per_node]:
            if absolute_weight < min_correlation:
                continue
            selected_pairs.add(tuple(sorted((left, right))))

    # Guarantee sleep-centered connectivity requested by user.
    for feature in columns:
        if feature == focus_feature:
            continue
        selected_pairs.add(tuple(sorted((focus_feature, feature))))

    edges = []
    for left, right in selected_pairs:
        correlation_value = float(correlations.loc[left, right])
        edges.append(
            {
                "from": left,
                "to": right,
                "value": correlation_value,
                "abs_value": abs(correlation_value),
            }
        )

    edges.sort(key=lambda edge: (
        edge["abs_value"], edge["from"], edge["to"]), reverse=True)
    return edges[:max_edges]


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Extended dataset not found: {data_path}")

    frame = pd.read_csv(data_path)
    if args.color not in frame.columns:
        raise ValueError(f"Focus feature '{args.color}' not found in dataset.")

    numeric_columns = [
        column for column in frame.columns if column != "record_id" and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if len(numeric_columns) < 2:
        raise ValueError(
            "Dataset must contain at least two numeric feature columns.")

    for axis_feature in (args.x, args.y, args.z):
        if axis_feature not in numeric_columns:
            raise ValueError(
                f"3D axis feature '{axis_feature}' not found in numeric columns.")

    correlations = frame[numeric_columns].corr(method="pearson").fillna(0.0)
    edges = build_edges(
        correlations=correlations,
        focus_feature=args.color,
        min_correlation=args.min_correlation,
        top_edges_per_node=args.top_edges_per_node,
        max_edges=args.max_edges,
    )

    coords: dict[str, tuple[float, float, float]] = {}
    nodes = []
    for feature in numeric_columns:
        corr_focus = 1.0 if feature == args.color else float(
            correlations.loc[args.color, feature])
        x_coord = float(correlations.loc[feature, args.x])
        y_coord = float(correlations.loc[feature, args.y])
        z_coord = float(correlations.loc[feature, args.z])
        if feature == args.color:
            x_coord += 0.08
            y_coord += 0.08
            z_coord += 0.08
        coords[feature] = (x_coord, y_coord, z_coord)
        nodes.append(
            {
                "id": feature,
                "group": classify_feature(feature, args.color),
                "label": display_name(feature),
                "size": round(9 + 18 * abs(corr_focus), 3),
                "hover": (
                    f"{display_name(feature)}<br>"
                    f"Correlation with {display_name(args.color)}: {corr_focus:.3f}<br>"
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

    edge_payload = []
    for edge in edges:
        left = str(edge["from"])
        right = str(edge["to"])
        value = float(edge["value"])
        lx, ly, lz = coords[left]
        rx, ry, rz = coords[right]

        if value >= 0:
            positive_x.extend([lx, rx, None])
            positive_y.extend([ly, ry, None])
            positive_z.extend([lz, rz, None])
        else:
            negative_x.extend([lx, rx, None])
            negative_y.extend([ly, ry, None])
            negative_z.extend([lz, rz, None])

        edge_payload.append(
            {"from": left, "to": right, "value": value, "abs_value": abs(value)})

    top_to_sleep = []
    for feature in numeric_columns:
        if feature == args.color:
            continue
        top_to_sleep.append(
            {
                "label": display_name(feature),
                "correlation": float(correlations.loc[args.color, feature]),
            }
        )
    top_to_sleep.sort(key=lambda item: abs(item["correlation"]), reverse=True)

    group_colors = {
        "sleep": "#7ea37f",
        "lifestyle": "#c98d54",
        "derived": "#2a6f97",
        "other": "#9f9aa4",
    }

    payload = {
        "subtitle": (
            "3D correlation graph from your dataset. Every feature node is labeled and connected to sleep quality score. "
            "Blue lines are positive, red dotted lines are negative."
        ),
        "nodes_x": [float(node["x"]) for node in nodes],
        "nodes_y": [float(node["y"]) for node in nodes],
        "nodes_z": [float(node["z"]) for node in nodes],
        "nodes_text": [str(node["label"]) for node in nodes],
        "node_hover": [str(node["hover"]) for node in nodes],
        "node_sizes": [float(node["size"]) for node in nodes],
        "node_colors": [group_colors.get(str(node["group"]), "#9f9aa4") for node in nodes],
        "positive_x": positive_x,
        "positive_y": positive_y,
        "positive_z": positive_z,
        "negative_x": negative_x,
        "negative_y": negative_y,
        "negative_z": negative_z,
        "focus_label": display_name(args.color),
        "feature_count": len(nodes),
        "edge_count": len(edge_payload),
        "min_correlation": args.min_correlation,
        "top_to_sleep": top_to_sleep,
        "axis_labels": {
            "x": display_name(args.x),
            "y": display_name(args.y),
            "z": display_name(args.z),
        },
    }

    html_template = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Sleep Feature Correlation Network (3D)</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    html, body { margin: 0; min-height: 100%; background: #f5f1e8; color: #1f1e1a; font-family: Georgia, 'Times New Roman', serif; }
    .page { max-width: 1500px; margin: 0 auto; padding: 20px; }
    .layout { display: grid; grid-template-columns: minmax(300px, 360px) minmax(0, 1fr); gap: 16px; }
    .panel { background: rgba(255,252,246,0.96); border: 1px solid rgba(75,68,52,0.12); border-radius: 14px; box-shadow: 0 10px 30px rgba(72,61,45,0.08); }
    .left { padding: 14px; }
    #network { height: 82vh; min-height: 700px; }
    .chip { margin: 8px 0; padding: 9px 10px; border-radius: 10px; background: rgba(245,240,231,0.9); border: 1px solid rgba(75,68,52,0.12); }
    .list { margin: 10px 0 0 0; padding: 0; list-style: none; }
    .list li { display: flex; justify-content: space-between; gap: 8px; padding: 7px 0; border-bottom: 1px solid rgba(80,74,64,0.09); font-size: 14px; }
    @media (max-width: 1080px) { .layout { grid-template-columns: 1fr; } #network { min-height: 620px; height: 72vh; } }
  </style>
</head>
<body>
  <div class=\"page\">
    <h1 style=\"margin: 0 0 8px 0;\">Sleep Feature Correlation Network (3D)</h1>
    <p id=\"subtitle\" style=\"margin: 0 0 14px 0; color: #696458;\"></p>
    <div class=\"layout\">
      <aside class=\"panel left\">
        <div class=\"chip\"><strong id=\"feature-count\"></strong> labeled feature nodes</div>
        <div class=\"chip\"><strong id=\"edge-count\"></strong> correlation links</div>
        <div class=\"chip\">Min feature-feature threshold: <strong id=\"min-correlation\"></strong></div>
        <div class=\"chip\">Focus node: <strong id=\"focus-name\"></strong></div>
        <h3 style=\"margin: 12px 0 6px 0;\">Top Correlations To Focus</h3>
        <ul id=\"top-list\" class=\"list\"></ul>
      </aside>
      <section class=\"panel\">
        <div id=\"network\"></div>
      </section>
    </div>
  </div>
  <script>
    const payload = __PAYLOAD__;
    document.getElementById('subtitle').textContent = payload.subtitle;
    document.getElementById('feature-count').textContent = payload.feature_count;
    document.getElementById('edge-count').textContent = payload.edge_count;
    document.getElementById('min-correlation').textContent = payload.min_correlation.toFixed(2);
    document.getElementById('focus-name').textContent = payload.focus_label;

    const list = document.getElementById('top-list');
    payload.top_to_sleep.slice(0, 10).forEach((entry) => {
      const li = document.createElement('li');
      const left = document.createElement('span');
      left.textContent = entry.label;
      const right = document.createElement('span');
      right.textContent = entry.correlation.toFixed(3);
      li.appendChild(left);
      li.appendChild(right);
      list.appendChild(li);
    });

    const posEdges = {
      type: 'scatter3d', mode: 'lines', x: payload.positive_x, y: payload.positive_y, z: payload.positive_z,
      hoverinfo: 'skip', line: { color: 'rgba(42,111,151,0.58)', width: 5 }, name: 'Positive'
    };

    const negEdges = {
      type: 'scatter3d', mode: 'lines', x: payload.negative_x, y: payload.negative_y, z: payload.negative_z,
      hoverinfo: 'skip', line: { color: 'rgba(184,74,58,0.58)', width: 5, dash: 'dot' }, name: 'Negative'
    };

    const nodes = {
      type: 'scatter3d', mode: 'markers+text', x: payload.nodes_x, y: payload.nodes_y, z: payload.nodes_z,
      text: payload.nodes_text, textposition: 'top center', textfont: { size: 14, color: '#1f1e1a' },
      hovertext: payload.node_hover, hoverinfo: 'text',
      marker: { size: payload.node_sizes, color: payload.node_colors, opacity: 0.95, line: { color: 'rgba(255,255,255,0.86)', width: 1 } },
      name: 'Features'
    };

    const layout = {
      margin: { l: 0, r: 0, b: 0, t: 0 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      scene: {
        xaxis: { title: payload.axis_labels.x, range: [-1.15, 1.15] },
        yaxis: { title: payload.axis_labels.y, range: [-1.15, 1.15] },
        zaxis: { title: payload.axis_labels.z, range: [-1.15, 1.15] },
        camera: { eye: { x: 1.45, y: 1.35, z: 1.3 } }
      },
      legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(255,252,246,0.72)' }
    };

    Plotly.newPlot('network', [posEdges, negEdges, nodes], layout, { responsive: true, displaylogo: false });
  </script>
</body>
</html>
"""

    html = html_template.replace("__PAYLOAD__", json.dumps(payload))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print("Sleep feature correlation network (3D) complete.")
    print(f"Features plotted: {len(nodes)}")
    print(f"Edges plotted: {len(edge_payload)}")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
