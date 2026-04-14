from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleep_model.analysis import safe_correlation, threshold_sweep_auroc, train_test_indices  # noqa: E402
from sleep_model.autoencoder import NumpyAutoencoder  # noqa: E402
from sleep_model.data_processing import load_and_process_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run quant-to-qual extension analyses: per-feature error, weight sensitivity, and latent export."
    )
    parser.add_argument("--schema", default=str(ROOT /
                        "config" / "feature_schema.json"))
    parser.add_argument(
        "--sleep-edf-data", default=str(ROOT / "data" / "sleep_edfx_model_input.csv"))
    parser.add_argument("--cap-data", default=str(ROOT /
                        "data" / "capslpdb_model_input.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-rates", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--latent-export-mask-rate", type=float, default=0.3)
    return parser.parse_args()


def parse_mask_rates(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one mask rate is required.")
    for value in values:
        if value < 0.0 or value >= 1.0:
            raise ValueError("Mask rates must be in [0.0, 1.0).")
    return values


def apply_feature_mask(matrix: np.ndarray, mask_rate: float, rng: np.random.Generator) -> np.ndarray:
    masked = matrix.copy()
    feature_mask = rng.random(masked.shape) < mask_rate
    masked[feature_mask] = 0.0
    return masked


def train_quant_to_qual_model(
    train_input: np.ndarray,
    train_target: np.ndarray,
    test_input: np.ndarray,
    test_target: np.ndarray,
    mask_rate: float,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> NumpyAutoencoder:
    rng = np.random.default_rng(seed)
    masked_train_input = apply_feature_mask(train_input, mask_rate, rng)
    model = NumpyAutoencoder(
        input_dim=train_input.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        output_dim=train_target.shape[1],
        seed=seed,
    )
    model.fit(
        train_features=masked_train_input,
        target_train_features=train_target,
        val_features=test_input,
        target_val_features=test_target,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    return model


def perturb_weights(weights: np.ndarray, feature_index: int, delta: float) -> np.ndarray:
    current = float(weights[feature_index])
    target = float(np.clip(current + delta, 1e-6, 1.0 - 1e-6))
    if np.isclose(target, current):
        return np.array(weights, copy=True)

    other_indices = [index for index in range(
        len(weights)) if index != feature_index]
    remaining_sum = float(np.sum(weights[other_indices]))
    new_remaining_sum = 1.0 - target
    if remaining_sum <= 0.0 or new_remaining_sum <= 0.0:
        return np.array(weights, copy=True)

    scale = new_remaining_sum / remaining_sum
    perturbed = np.array(weights, copy=True)
    perturbed[feature_index] = target
    perturbed[other_indices] = perturbed[other_indices] * scale
    perturbed = np.clip(perturbed, 1e-6, None)
    perturbed = perturbed / np.sum(perturbed)
    return perturbed


def dataset_name_from_path(path: str | Path) -> str:
    stem = Path(path).stem.lower()
    if "sleep" in stem:
        return "sleepedfx"
    if "cap" in stem:
        return "capslpdb"
    return stem


def run_extension_analysis(dataset_path: str, args: argparse.Namespace, mask_rates: list[float]) -> dict[str, object]:
    dataset = load_and_process_dataset(dataset_path, args.schema)
    train_idx, test_idx = train_test_indices(
        len(dataset.rows), test_ratio=args.test_ratio, seed=args.seed)

    quant_indices = [index for index, kind in enumerate(
        dataset.feature_types) if kind == "quantitative"]
    qual_indices = [index for index, kind in enumerate(
        dataset.feature_types) if kind == "qualitative"]
    qual_feature_names = [dataset.feature_names[index]
                          for index in qual_indices]

    input_matrix = dataset.processed_matrix[:, quant_indices]
    target_matrix = dataset.processed_matrix[:, qual_indices]
    qual_weights = dataset.weights[qual_indices]
    qual_weights = qual_weights / np.sum(qual_weights)

    train_input = input_matrix[train_idx]
    test_input = input_matrix[test_idx]
    train_target = target_matrix[train_idx]
    test_target = target_matrix[test_idx]
    true_qual_scores = test_target @ qual_weights

    per_feature_rows: list[dict[str, object]] = []
    for mask_rate in mask_rates:
        model = train_quant_to_qual_model(
            train_input,
            train_target,
            test_input,
            test_target,
            mask_rate=mask_rate,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        predicted_qual = model.reconstruct(test_input)
        predicted_qual_scores = predicted_qual @ qual_weights
        mean_auroc, _ = threshold_sweep_auroc(
            true_qual_scores, predicted_qual_scores)

        mse_by_feature = {}
        for offset, feature_name in enumerate(qual_feature_names):
            mse_by_feature[feature_name] = float(
                np.mean((predicted_qual[:, offset] - test_target[:, offset]) ** 2))

        per_feature_rows.append(
            {
                "mask_rate": float(mask_rate),
                "mean_threshold_sweep_auroc": mean_auroc,
                "score_correlation_test": safe_correlation(true_qual_scores, predicted_qual_scores),
                "reconstruction_mse_overall": float(np.mean((predicted_qual - test_target) ** 2)),
                "reconstruction_mse_by_feature": mse_by_feature,
            }
        )

    average_mse_by_feature = {}
    for feature_name in qual_feature_names:
        average_mse_by_feature[feature_name] = float(
            np.mean([row["reconstruction_mse_by_feature"][feature_name]
                    for row in per_feature_rows])
        )
    hardest_features = [
        {"feature": name, "average_mse": value}
        for name, value in sorted(average_mse_by_feature.items(), key=lambda item: item[1], reverse=True)
    ]

    baseline_model = train_quant_to_qual_model(
        train_input,
        train_target,
        test_input,
        test_target,
        mask_rate=0.0,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    predicted_qual_baseline = baseline_model.reconstruct(test_input)

    true_full_test = np.array(dataset.processed_matrix[test_idx], copy=True)
    predicted_full_test = np.array(true_full_test, copy=True)
    predicted_full_test[:, qual_indices] = predicted_qual_baseline

    baseline_true_scores = true_full_test @ dataset.weights
    baseline_pred_scores = predicted_full_test @ dataset.weights
    baseline_auroc, _ = threshold_sweep_auroc(
        baseline_true_scores, baseline_pred_scores)

    sensitivity_rows: list[dict[str, float | str]] = []
    for feature_index, feature_name in enumerate(dataset.feature_names):
        auroc_delta_minus = 0.0
        auroc_delta_plus = 0.0

        for delta in (-0.05, 0.05):
            perturbed_weights = perturb_weights(
                dataset.weights, feature_index, delta)
            true_scores = true_full_test @ perturbed_weights
            predicted_scores = predicted_full_test @ perturbed_weights
            perturbed_auroc, _ = threshold_sweep_auroc(
                true_scores, predicted_scores)
            delta_auroc = float(perturbed_auroc - baseline_auroc)

            if delta < 0:
                auroc_delta_minus = delta_auroc
            else:
                auroc_delta_plus = delta_auroc

        sensitivity_rows.append(
            {
                "feature": feature_name,
                "auroc_delta_minus_0_05": auroc_delta_minus,
                "auroc_delta_plus_0_05": auroc_delta_plus,
                "max_abs_auroc_delta": float(max(abs(auroc_delta_minus), abs(auroc_delta_plus))),
            }
        )

    sensitivity_rows = sorted(
        sensitivity_rows,
        key=lambda row: float(row["max_abs_auroc_delta"]),
        reverse=True,
    )

    output = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "row_count": len(dataset.rows),
        "test_ratio": args.test_ratio,
        "mask_rates": mask_rates,
        "evaluation_mode": "quant_to_qual",
        "input_feature_names": [dataset.feature_names[index] for index in quant_indices],
        "target_feature_names": qual_feature_names,
        "per_mask_feature_errors": per_feature_rows,
        "hardest_qualitative_features": hardest_features,
        "weight_sensitivity": {
            "baseline_auroc": float(baseline_auroc),
            "rows": sensitivity_rows,
        },
    }

    if dataset_name_from_path(dataset_path) == "sleepedfx":
        latent_mask_rate = args.latent_export_mask_rate
        latent_model = train_quant_to_qual_model(
            train_input,
            train_target,
            test_input,
            test_target,
            mask_rate=latent_mask_rate,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        latent_coords = latent_model.encode(test_input)
        predicted_qual = latent_model.reconstruct(test_input)
        predicted_qual_scores = predicted_qual @ qual_weights
        true_full_scores = dataset.quality_scores[test_idx]

        output["latent_export"] = {
            "mask_rate": latent_mask_rate,
            "row_count": int(len(test_idx)),
        }

        latent_rows = []
        for local_index, source_index in enumerate(test_idx):
            latent_rows.append(
                {
                    "source_row_index": int(source_index),
                    "latent_x": float(latent_coords[local_index, 0]),
                    "latent_y": float(latent_coords[local_index, 1]),
                    "true_sleep_quality_score": float(true_full_scores[local_index]),
                    "predicted_qual_score": float(predicted_qual_scores[local_index]),
                }
            )
        output["latent_rows"] = latent_rows

    return output


def save_sleep_latent_csv(output: dict[str, object], csv_path: Path) -> None:
    rows = output.get("latent_rows")
    if not isinstance(rows, list):
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_row_index",
                "latent_x",
                "latent_y",
                "true_sleep_quality_score",
                "predicted_qual_score",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_hardest_feature_table(dataset_label: str, output: dict[str, object]) -> None:
    rows = output["hardest_qualitative_features"]
    print(
        f"\n{dataset_label} hardest qualitative features (avg MSE across mask rates):")
    print(f"{'rank':>4} {'feature':>24} {'avg_mse':>12}")
    print("-" * 44)
    for rank, row in enumerate(rows, start=1):
        print(f"{rank:>4} {row['feature']:>24} {row['average_mse']:>12.6f}")


def print_weight_sensitivity_table(dataset_label: str, output: dict[str, object]) -> None:
    sensitivity = output["weight_sensitivity"]
    rows = sensitivity["rows"]
    print(f"\n{dataset_label} weight sensitivity (AUROC delta from baseline):")
    print(f"baseline_auroc={sensitivity['baseline_auroc']:.6f}")
    print(f"{'rank':>4} {'feature':>24} {'delta_-0.05':>12} {'delta_+0.05':>12} {'max_abs':>12}")
    print("-" * 70)
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:>4} {row['feature']:>24} "
            f"{row['auroc_delta_minus_0_05']:>12.6f} "
            f"{row['auroc_delta_plus_0_05']:>12.6f} "
            f"{row['max_abs_auroc_delta']:>12.6f}"
        )


def main() -> None:
    args = parse_args()
    mask_rates = parse_mask_rates(args.mask_rates)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("Sleep-EDF", args.sleep_edf_data, output_dir /
         "quant_to_qual_extensions_sleepedfx.json"),
        ("CAP", args.cap_data, output_dir /
         "quant_to_qual_extensions_capslpdb.json"),
    ]

    for label, dataset_path, json_path in datasets:
        output = run_extension_analysis(dataset_path, args, mask_rates)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)

        print_hardest_feature_table(label, output)
        print_weight_sensitivity_table(label, output)

        if label == "Sleep-EDF":
            latent_csv_path = output_dir / "sleepedfx_quant_to_qual_latent_test.csv"
            save_sleep_latent_csv(output, latent_csv_path)
            print(
                f"\nSleep-EDF latent CSV saved to: {latent_csv_path.resolve()}")

        print(f"\nSaved extension analysis JSON to: {json_path.resolve()}")


if __name__ == "__main__":
    main()
