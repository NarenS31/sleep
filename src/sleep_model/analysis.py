from __future__ import annotations

import numpy as np

from sleep_model.data_processing import ProcessedDataset, normalize_feature_matrix


def train_test_indices(row_count: int, test_ratio: float = 0.25, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(row_count)
    split_index = int(row_count * (1.0 - test_ratio))
    train_idx = indices[:split_index]
    test_idx = indices[split_index:]
    return train_idx, test_idx


def roc_auc_score_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = labels == 1
    negatives = labels == 0

    positive_count = int(np.sum(positives))
    negative_count = int(np.sum(negatives))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC AUC requires both positive and negative labels.")

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    true_positive_rate = np.cumsum(sorted_labels == 1) / positive_count
    false_positive_rate = np.cumsum(sorted_labels == 0) / negative_count

    tpr = np.concatenate(([0.0], true_positive_rate, [1.0]))
    fpr = np.concatenate(([0.0], false_positive_rate, [1.0]))
    return float(np.trapezoid(tpr, fpr))


def threshold_sweep_auroc(
    original_scores: np.ndarray,
    reconstructed_scores: np.ndarray,
    quantiles: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65),
) -> tuple[float, dict[str, float]]:
    auc_by_threshold: dict[str, float] = {}
    auc_values: list[float] = []

    for quantile in quantiles:
        threshold = float(np.quantile(original_scores, quantile))
        labels = (original_scores >= threshold).astype(int)
        if len(np.unique(labels)) < 2:
            continue
        auc = roc_auc_score_binary(labels, reconstructed_scores)
        auc_by_threshold[f"{quantile:.2f}"] = auc
        auc_values.append(auc)

    if not auc_values:
        raise ValueError("Unable to compute AUROC because all threshold splits collapsed to one class.")

    return float(np.mean(auc_values)), auc_by_threshold


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def monte_carlo_stability(
    processed_matrix: np.ndarray,
    weights: np.ndarray,
    iterations: int = 2000,
    feature_noise: float = 0.05,
    weight_noise: float = 0.02,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    base_scores = processed_matrix @ weights
    correlations: list[float] = []
    score_deviations: list[float] = []

    for _ in range(iterations):
        noisy_features = np.clip(processed_matrix + rng.normal(0.0, feature_noise, processed_matrix.shape), 0.0, 1.0)
        noisy_weights = np.clip(weights + rng.normal(0.0, weight_noise, weights.shape), 1e-6, None)
        noisy_weights = noisy_weights / noisy_weights.sum()

        simulated_scores = noisy_features @ noisy_weights
        correlations.append(safe_correlation(base_scores, simulated_scores))
        score_deviations.append(float(np.std(simulated_scores - base_scores)))

    return {
        "mean_score_correlation": float(np.mean(correlations)),
        "score_correlation_std": float(np.std(correlations)),
        "mean_score_deviation": float(np.mean(score_deviations)),
        "score_deviation_std": float(np.std(score_deviations)),
    }


def monte_carlo_qualitative_simulation(
    dataset: ProcessedDataset,
    iterations: int = 5000,
    qualitative_variance: float = 0.5,
    quantitative_noise: float = 0.02,
    weight_noise: float = 0.03,
    seed: int = 42,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    base_scores = dataset.quality_scores
    qualitative_indices = [index for index, kind in enumerate(dataset.feature_types) if kind == "qualitative"]
    quantitative_indices = [index for index, kind in enumerate(dataset.feature_types) if kind == "quantitative"]

    correlations: list[float] = []
    score_deviations: list[float] = []
    mean_scores: list[float] = []
    aurocs: list[float] = []
    simulated_weight_rows: list[np.ndarray] = []
    qualitative_shift_rows: list[np.ndarray] = []

    for _ in range(iterations):
        simulated_raw = np.array(dataset.raw_numeric_matrix, copy=True)

        if quantitative_indices:
            simulated_raw[:, quantitative_indices] += rng.normal(
                0.0,
                quantitative_noise,
                size=(simulated_raw.shape[0], len(quantitative_indices)),
            ) * (dataset.feature_maximums[quantitative_indices] - dataset.feature_minimums[quantitative_indices])

        if qualitative_indices:
            qualitative_shifts = rng.uniform(
                -qualitative_variance,
                qualitative_variance,
                size=(simulated_raw.shape[0], len(qualitative_indices)),
            )
            simulated_raw[:, qualitative_indices] += qualitative_shifts
            qualitative_shift_rows.append(np.mean(qualitative_shifts, axis=0))

        simulated_raw = np.clip(simulated_raw, dataset.feature_minimums, dataset.feature_maximums)

        simulated_processed = normalize_feature_matrix(
            simulated_raw,
            dataset.feature_minimums,
            dataset.feature_maximums,
            dataset.feature_directions,
        )

        simulated_weights = np.clip(dataset.weights + rng.normal(0.0, weight_noise, dataset.weights.shape), 1e-6, None)
        simulated_weights = simulated_weights / simulated_weights.sum()
        simulated_weight_rows.append(simulated_weights)

        simulated_scores = simulated_processed @ simulated_weights
        mean_auroc, _ = threshold_sweep_auroc(base_scores, simulated_scores)

        correlations.append(safe_correlation(base_scores, simulated_scores))
        score_deviations.append(float(np.std(simulated_scores - base_scores)))
        mean_scores.append(float(np.mean(simulated_scores)))
        aurocs.append(mean_auroc)

    weight_matrix = np.vstack(simulated_weight_rows)
    qualitative_shift_matrix = (
        np.vstack(qualitative_shift_rows)
        if qualitative_shift_rows
        else np.zeros((iterations, 0), dtype=float)
    )

    feature_weight_summary = {
        dataset.feature_names[index]: {
            "mean_weight": float(np.mean(weight_matrix[:, index])),
            "weight_std": float(np.std(weight_matrix[:, index])),
        }
        for index in range(weight_matrix.shape[1])
    }

    qualitative_shift_summary = {}
    for offset, index in enumerate(qualitative_indices):
        qualitative_shift_summary[dataset.feature_names[index]] = {
            "mean_shift": float(np.mean(qualitative_shift_matrix[:, offset])),
            "shift_std": float(np.std(qualitative_shift_matrix[:, offset])),
        }

    return {
        "iterations": iterations,
        "qualitative_variance": qualitative_variance,
        "quantitative_noise": quantitative_noise,
        "weight_noise": weight_noise,
        "mean_score_correlation": float(np.mean(correlations)),
        "score_correlation_std": float(np.std(correlations)),
        "mean_score_deviation": float(np.mean(score_deviations)),
        "score_deviation_std": float(np.std(score_deviations)),
        "mean_simulated_score": float(np.mean(mean_scores)),
        "simulated_score_std": float(np.std(mean_scores)),
        "mean_auroc": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "feature_weight_summary": feature_weight_summary,
        "qualitative_shift_summary": qualitative_shift_summary,
        "auroc_samples": [float(value) for value in aurocs],
        "score_correlation_samples": [float(value) for value in correlations],
        "mean_score_samples": [float(value) for value in mean_scores],
    }


def feature_contributions(processed_matrix: np.ndarray, weights: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    contributions = processed_matrix.mean(axis=0) * weights
    return {feature_names[index]: float(value) for index, value in enumerate(contributions)}
