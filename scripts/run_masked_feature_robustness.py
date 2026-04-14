from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sleep_model.analysis import safe_correlation, threshold_sweep_auroc, train_test_indices  # noqa: E402
from sleep_model.autoencoder import NumpyAutoencoder  # noqa: E402
from sleep_model.data_processing import load_and_process_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run masked-feature denoising robustness experiments for the sleep autoencoder."
    )
    parser.add_argument("--data", default=str(ROOT /
                        "data" / "all_datasets_model_input.csv"))
    parser.add_argument("--schema", default=str(ROOT /
                        "config" / "feature_schema.json"))
    parser.add_argument("--output", default=str(ROOT /
                        "outputs" / "masked_feature_robustness.json"))
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mask-rates",
        default="0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated mask rates applied to the full stacked input matrix.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=["full_reconstruction", "quant_to_qual"],
        default="quant_to_qual",
        help="Evaluation design: reconstruct all stacked features or predict held-out qualitative features from quantitative inputs.",
    )
    return parser.parse_args()


def parse_mask_rates(text: str) -> list[float]:
    rates = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not rates:
        raise ValueError("At least one mask rate is required.")
    for value in rates:
        if value < 0.0 or value >= 1.0:
            raise ValueError("Mask rates must be in [0.0, 1.0).")
    return rates


def apply_feature_mask(matrix: np.ndarray, mask_rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    masked = matrix.copy()
    feature_mask = rng.random(masked.shape) < mask_rate
    masked[feature_mask] = 0.0
    return masked, feature_mask


def run_full_reconstruction_mode(
    dataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    mask_rates: list[float],
    args: argparse.Namespace,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    quant_indices = [index for index, kind in enumerate(
        dataset.feature_types) if kind == "quantitative"]
    quantitative_raw = dataset.raw_numeric_matrix[:, quant_indices]
    quantitative_processed = dataset.processed_matrix[:, quant_indices]
    quantitative_weights = dataset.weights[quant_indices]
    quantitative_weights = quantitative_weights / quantitative_weights.sum()
    quantitative_base_scores = quantitative_processed @ quantitative_weights

    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=220,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=args.seed,
    )
    xgb_model.fit(
        quantitative_raw[train_idx],
        quantitative_base_scores[train_idx],
        eval_set=[(quantitative_raw[test_idx],
                   quantitative_base_scores[test_idx])],
        verbose=False,
    )

    quantitative_proxy_scores = np.clip(
        xgb_model.predict(quantitative_raw), 0.0, 1.0)
    stacked_matrix = np.column_stack(
        (dataset.processed_matrix, quantitative_proxy_scores))
    schema_feature_count = len(dataset.feature_names)

    train_target = stacked_matrix[train_idx]
    test_target = stacked_matrix[test_idx]
    original_scores_test = dataset.quality_scores[test_idx]

    rng = np.random.default_rng(args.seed)
    experiment_rows: list[dict[str, float]] = []
    for mask_rate in mask_rates:
        train_masked, _ = apply_feature_mask(train_target, mask_rate, rng)

        model = NumpyAutoencoder(
            input_dim=train_masked.shape[1],
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            seed=args.seed,
        )
        model.fit(
            train_features=train_masked,
            target_train_features=train_target,
            target_val_features=test_target,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )

        reconstructed = model.reconstruct(test_target)
        reconstructed_matrix = reconstructed[:, :schema_feature_count]
        target_matrix = test_target[:, :schema_feature_count]
        reconstructed_scores = reconstructed_matrix @ dataset.weights

        mean_auroc, threshold_results = threshold_sweep_auroc(
            original_scores_test, reconstructed_scores)
        full_mse = float(np.mean((reconstructed_matrix - target_matrix) ** 2))
        score_corr = safe_correlation(
            original_scores_test, reconstructed_scores)

        experiment_rows.append(
            {
                "mask_rate": float(mask_rate),
                "mean_threshold_sweep_auroc": mean_auroc,
                "reconstruction_mse_full": full_mse,
                "score_correlation_test": score_corr,
                "threshold_auroc_0_35": float(threshold_results.get("0.35", np.nan)),
                "threshold_auroc_0_45": float(threshold_results.get("0.45", np.nan)),
                "threshold_auroc_0_55": float(threshold_results.get("0.55", np.nan)),
                "threshold_auroc_0_65": float(threshold_results.get("0.65", np.nan)),
            }
        )

    metadata = {
        "feature_count": schema_feature_count,
        "feature_names": dataset.feature_names,
        "stacked_feature_count": int(stacked_matrix.shape[1]),
        "stacked_feature_names": dataset.feature_names + ["xgb_quantitative_score"],
        "weights": {name: float(dataset.weights[index]) for index, name in enumerate(dataset.feature_names)},
        "evaluation": {
            "mode": "full_reconstruction",
            "training_inputs": "masked_stacked_features",
            "training_targets": "clean_stacked_features",
            "inference_inputs": "clean_stacked_features",
            "score_features": "schema feature columns only",
        },
    }
    return experiment_rows, metadata


def run_quant_to_qual_mode(
    dataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    mask_rates: list[float],
    args: argparse.Namespace,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    quant_indices = [index for index, kind in enumerate(
        dataset.feature_types) if kind == "quantitative"]
    qual_indices = [index for index, kind in enumerate(
        dataset.feature_types) if kind == "qualitative"]

    input_matrix = dataset.processed_matrix[:, quant_indices]
    target_matrix = dataset.processed_matrix[:, qual_indices]
    qual_weights = dataset.weights[qual_indices]
    qual_weights = qual_weights / qual_weights.sum()

    train_input = input_matrix[train_idx]
    test_input = input_matrix[test_idx]
    train_target = target_matrix[train_idx]
    test_target = target_matrix[test_idx]
    true_scores_test = test_target @ qual_weights

    rng = np.random.default_rng(args.seed)
    experiment_rows: list[dict[str, float]] = []
    for mask_rate in mask_rates:
        train_masked, _ = apply_feature_mask(train_input, mask_rate, rng)

        model = NumpyAutoencoder(
            input_dim=train_input.shape[1],
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            output_dim=train_target.shape[1],
            seed=args.seed,
        )
        model.fit(
            train_features=train_masked,
            target_train_features=train_target,
            val_features=test_input,
            target_val_features=test_target,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )

        predicted_qual = model.reconstruct(test_input)
        predicted_scores = predicted_qual @ qual_weights
        mean_auroc, threshold_results = threshold_sweep_auroc(
            true_scores_test, predicted_scores)
        full_mse = float(np.mean((predicted_qual - test_target) ** 2))
        score_corr = safe_correlation(true_scores_test, predicted_scores)

        experiment_rows.append(
            {
                "mask_rate": float(mask_rate),
                "mean_threshold_sweep_auroc": mean_auroc,
                "reconstruction_mse_full": full_mse,
                "score_correlation_test": score_corr,
                "threshold_auroc_0_35": float(threshold_results.get("0.35", np.nan)),
                "threshold_auroc_0_45": float(threshold_results.get("0.45", np.nan)),
                "threshold_auroc_0_55": float(threshold_results.get("0.55", np.nan)),
                "threshold_auroc_0_65": float(threshold_results.get("0.65", np.nan)),
            }
        )

    metadata = {
        "input_feature_count": len(quant_indices),
        "input_feature_names": [dataset.feature_names[index] for index in quant_indices],
        "target_feature_count": len(qual_indices),
        "target_feature_names": [dataset.feature_names[index] for index in qual_indices],
        "target_weights": {
            dataset.feature_names[index]: float(qual_weights[offset])
            for offset, index in enumerate(qual_indices)
        },
        "evaluation": {
            "mode": "quant_to_qual",
            "training_inputs": "masked_quantitative_features",
            "training_targets": "clean_qualitative_features",
            "inference_inputs": "clean_quantitative_features",
            "score_features": "qualitative target columns only",
        },
    }
    return experiment_rows, metadata


def main() -> None:
    args = parse_args()
    mask_rates = parse_mask_rates(args.mask_rates)

    dataset = load_and_process_dataset(args.data, args.schema)
    train_idx, test_idx = train_test_indices(
        len(dataset.rows), test_ratio=args.test_ratio, seed=args.seed)

    if args.evaluation_mode == "quant_to_qual":
        experiment_rows, mode_metadata = run_quant_to_qual_mode(
            dataset,
            train_idx,
            test_idx,
            mask_rates,
            args,
        )
    else:
        experiment_rows, mode_metadata = run_full_reconstruction_mode(
            dataset,
            train_idx,
            test_idx,
            mask_rates,
            args,
        )

    output = {
        "dataset_path": str(Path(args.data).resolve()),
        "row_count": len(dataset.rows),
        "test_ratio": args.test_ratio,
        "epochs": args.epochs,
        "evaluation_mode": args.evaluation_mode,
        "mask_rates": mask_rates,
        **mode_metadata,
        "experiments": experiment_rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print("Masked-feature robustness run complete.")
    print(f"Saved to: {output_path.resolve()}")
    for row in experiment_rows:
        print(
            f"mask={row['mask_rate']:.2f} | AUROC={row['mean_threshold_sweep_auroc']:.4f} "
            f"| full_mse={row['reconstruction_mse_full']:.5f}"
        )


if __name__ == "__main__":
    main()
