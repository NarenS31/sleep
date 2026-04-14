from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
sys.path.insert(0, str(ROOT / "src"))

from sleep_model.analysis import (  # noqa: E402
    feature_contributions,
    monte_carlo_qualitative_simulation,
    monte_carlo_stability,
    safe_correlation,
    threshold_sweep_auroc,
    train_test_indices,
)
from sleep_model.autoencoder import NumpyAutoencoder  # noqa: E402
from sleep_model.data_processing import load_and_process_dataset  # noqa: E402
try:  # noqa: E402
    from sleep_model.plots import (
        plot_correlation_heatmaps,
        plot_latent_space,
        plot_monte_carlo_distributions,
        plot_monte_carlo_stability,
        plot_score_alignment,
        plot_training_loss,
    )
    PLOTTING_AVAILABLE = True
except Exception as plot_import_error:  # noqa: E402
    PLOTTING_AVAILABLE = False
    PLOT_IMPORT_ERROR = plot_import_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the sleep-quality autoencoder project.")
    parser.add_argument("--data", default=str(ROOT /
                        "data" / "sample_sleep_data.csv"))
    parser.add_argument("--schema", default=str(ROOT /
                        "config" / "feature_schema.json"))
    parser.add_argument("--output", default=str(ROOT / "outputs"))
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--monte-carlo-iterations", type=int, default=800)
    parser.add_argument("--qualitative-variance", type=float, default=0.5)
    parser.add_argument("--quantitative-noise", type=float, default=0.02)
    parser.add_argument("--weight-noise", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_and_process_dataset(args.data, args.schema)
    train_idx, test_idx = train_test_indices(
        len(dataset.rows), test_ratio=args.test_ratio, seed=args.seed)

    quant_indices = [index for index, kind in enumerate(
        dataset.feature_types) if kind == "quantitative"]
    qual_indices = [index for index, kind in enumerate(
        dataset.feature_types) if kind == "qualitative"]

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

    stacked_feature_names = dataset.feature_names + ["xgb_quantitative_score"]
    stacked_matrix = np.column_stack(
        (dataset.processed_matrix, quantitative_proxy_scores))
    train_features = stacked_matrix[train_idx]
    test_features = stacked_matrix[test_idx]

    model = NumpyAutoencoder(
        input_dim=train_features.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )
    history = model.fit(
        train_features,
        val_features=test_features,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )

    reconstructed_stacked = model.reconstruct(stacked_matrix)
    reconstructed_matrix = reconstructed_stacked[:, : len(
        dataset.feature_names)]
    reconstructed_quantitative_proxy = reconstructed_stacked[:, -1]
    latent_matrix = model.encode(stacked_matrix)
    original_scores = dataset.quality_scores
    reconstructed_scores = reconstructed_matrix @ dataset.weights

    mean_auroc, threshold_results = threshold_sweep_auroc(
        original_scores[test_idx],
        reconstructed_scores[test_idx],
    )
    reconstruction_mse = float(np.mean(
        (reconstructed_matrix[test_idx] - dataset.processed_matrix[test_idx]) ** 2))
    score_correlation = safe_correlation(
        original_scores[test_idx], reconstructed_scores[test_idx])
    quantitative_stage_mse = float(
        np.mean(
            (quantitative_proxy_scores[test_idx] - quantitative_base_scores[test_idx]) ** 2)
    )
    quantitative_stage_correlation = safe_correlation(
        quantitative_base_scores[test_idx],
        quantitative_proxy_scores[test_idx],
    )
    robustness = monte_carlo_stability(
        dataset.processed_matrix,
        dataset.weights,
        iterations=args.monte_carlo_iterations,
        seed=args.seed,
    )
    monte_carlo_results = monte_carlo_qualitative_simulation(
        dataset,
        iterations=args.monte_carlo_iterations,
        qualitative_variance=args.qualitative_variance,
        quantitative_noise=args.quantitative_noise,
        weight_noise=args.weight_noise,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    noisy_features = np.clip(
        dataset.processed_matrix +
        rng.normal(0.0, 0.05, dataset.processed_matrix.shape),
        0.0,
        1.0,
    )
    noisy_weights = np.clip(
        dataset.weights + rng.normal(0.0, 0.02, dataset.weights.shape), 1e-6, None)
    noisy_weights = noisy_weights / noisy_weights.sum()
    simulated_scores = noisy_features @ noisy_weights

    if PLOTTING_AVAILABLE:
        plot_score_alignment(
            quantitative_base_scores,
            quantitative_proxy_scores,
            output_dir / "quantitative_stage_alignment.png",
        )
        plot_score_alignment(
            original_scores, reconstructed_scores, output_dir / "score_alignment.png")
        plot_correlation_heatmaps(
            dataset.processed_matrix,
            reconstructed_matrix,
            dataset.feature_names,
            output_dir / "correlation_heatmaps.png",
        )
        plot_latent_space(latent_matrix, original_scores,
                          output_dir / "latent_space.png")
        plot_training_loss(history, output_dir / "training_loss.png")
        plot_monte_carlo_stability(
            original_scores, simulated_scores, output_dir / "monte_carlo_stability.png")
        plot_monte_carlo_distributions(
            np.array(monte_carlo_results["auroc_samples"], dtype=float),
            np.array(
                monte_carlo_results["score_correlation_samples"], dtype=float),
            output_dir / "monte_carlo_distributions.png",
        )

    metrics = {
        "dataset_path": str(Path(args.data).resolve()),
        "row_count": len(dataset.rows),
        "feature_count": len(dataset.feature_names),
        "feature_names": dataset.feature_names,
        "stacked_feature_count": len(stacked_feature_names),
        "stacked_feature_names": stacked_feature_names,
        "weights": {name: float(dataset.weights[index]) for index, name in enumerate(dataset.feature_names)},
        "feature_contributions": feature_contributions(
            dataset.processed_matrix,
            dataset.weights,
            dataset.feature_names,
        ),
        "quantitative_stage": {
            "quantitative_feature_names": [dataset.feature_names[index] for index in quant_indices],
            "qualitative_feature_names": [dataset.feature_names[index] for index in qual_indices],
            "quantitative_score_mse_test": quantitative_stage_mse,
            "quantitative_score_correlation_test": quantitative_stage_correlation,
        },
        "reconstruction_mse_test": reconstruction_mse,
        "quantitative_proxy_reconstruction_mse_test": float(
            np.mean(
                (reconstructed_quantitative_proxy[test_idx] - quantitative_proxy_scores[test_idx]) ** 2)
        ),
        "score_correlation_test": score_correlation,
        "mean_threshold_sweep_auroc": mean_auroc,
        "threshold_auroc": threshold_results,
        "monte_carlo_basic": robustness,
        "monte_carlo_qualitative_scale": {
            key: value
            for key, value in monte_carlo_results.items()
            if not key.endswith("_samples")
        },
        "plotting_available": PLOTTING_AVAILABLE,
        "plotting_error": str(PLOT_IMPORT_ERROR) if not PLOTTING_AVAILABLE else None,
    }

    monte_carlo_samples_path = output_dir / "monte_carlo_samples.json"
    with monte_carlo_samples_path.open("w", encoding="utf-8") as handle:
        json.dump(monte_carlo_results, handle, indent=2)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Sleep model run complete.")
    print(f"Rows: {metrics['row_count']}")
    print(
        "Quantitative stage correlation: "
        f"{metrics['quantitative_stage']['quantitative_score_correlation_test']:.3f}"
    )
    print(
        f"Mean threshold-sweep AUROC: {metrics['mean_threshold_sweep_auroc']:.3f}")
    print(f"Test reconstruction MSE: {metrics['reconstruction_mse_test']:.4f}")
    print(f"Score correlation: {metrics['score_correlation_test']:.3f}")
    print(
        "Monte Carlo mean AUROC: "
        f"{metrics['monte_carlo_qualitative_scale']['mean_auroc']:.3f}"
    )
    print(f"Metrics saved to: {metrics_path}")
    if not PLOTTING_AVAILABLE:
        print("Plot generation skipped due to plotting import error.")


if __name__ == "__main__":
    main()
