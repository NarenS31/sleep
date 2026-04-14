from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _finalize_plot(path: str | Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_score_alignment(original_scores: np.ndarray, reconstructed_scores: np.ndarray, path: str | Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(original_scores, reconstructed_scores, alpha=0.7, edgecolor="none")
    lower = min(float(original_scores.min()), float(reconstructed_scores.min()))
    upper = max(float(original_scores.max()), float(reconstructed_scores.max()))
    plt.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.5, color="black")
    plt.xlabel("Original sleep score")
    plt.ylabel("Reconstructed sleep score")
    plt.title("Original vs. Reconstructed Sleep Scores")
    _finalize_plot(path)


def plot_correlation_heatmaps(
    original_matrix: np.ndarray,
    reconstructed_matrix: np.ndarray,
    feature_names: list[str],
    path: str | Path,
) -> None:
    original_corr = np.corrcoef(original_matrix, rowvar=False)
    reconstructed_corr = np.corrcoef(reconstructed_matrix, rowvar=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, matrix, title in (
        (axes[0], original_corr, "Original Feature Correlations"),
        (axes[1], reconstructed_corr, "Reconstructed Feature Correlations"),
    ):
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axis.set_title(title)
        axis.set_xticks(range(len(feature_names)))
        axis.set_xticklabels(feature_names, rotation=90, fontsize=8)
        axis.set_yticks(range(len(feature_names)))
        axis.set_yticklabels(feature_names, fontsize=8)

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Correlation")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_latent_space(latent_matrix: np.ndarray, scores: np.ndarray, path: str | Path) -> None:
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(latent_matrix[:, 0], latent_matrix[:, 1], c=scores, cmap="viridis", alpha=0.8)
    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.title("Latent Space Learned by the Autoencoder")
    plt.colorbar(scatter, label="Sleep quality score")
    _finalize_plot(path)


def plot_training_loss(history: dict[str, list[float]], path: str | Path) -> None:
    plt.figure(figsize=(7, 5))
    if history["train_loss"]:
        train_x = np.arange(len(history["train_loss"])) * 25
        plt.plot(train_x, history["train_loss"], label="Train loss", linewidth=2)
    if history["val_loss"]:
        val_x = np.arange(len(history["val_loss"])) * 25
        plt.plot(val_x, history["val_loss"], label="Validation loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared reconstruction loss")
    plt.title("Autoencoder Training Curve")
    plt.legend()
    _finalize_plot(path)


def plot_monte_carlo_stability(base_scores: np.ndarray, simulated_scores: np.ndarray, path: str | Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(simulated_scores - base_scores, bins=30, color="#4C72B0", alpha=0.85)
    plt.xlabel("Simulated score - base score")
    plt.ylabel("Frequency")
    plt.title("Monte Carlo Score Stability")
    _finalize_plot(path)


def plot_monte_carlo_distributions(
    auc_samples: np.ndarray,
    correlation_samples: np.ndarray,
    path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(auc_samples, bins=30, color="#55A868", alpha=0.85)
    axes[0].set_title("Monte Carlo AUROC Distribution")
    axes[0].set_xlabel("AUROC")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(correlation_samples, bins=30, color="#C44E52", alpha=0.85)
    axes[1].set_title("Monte Carlo Score Correlation Distribution")
    axes[1].set_xlabel("Correlation with base score")
    axes[1].set_ylabel("Frequency")

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
