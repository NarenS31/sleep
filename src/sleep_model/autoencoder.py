from __future__ import annotations

import numpy as np


class NumpyAutoencoder:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 8,
        latent_dim: int = 2,
        output_dim: int | None = None,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim
        self.W1 = rng.normal(0.0, np.sqrt(
            2.0 / (input_dim + hidden_dim)), size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0.0, np.sqrt(
            2.0 / (hidden_dim + latent_dim)), size=(hidden_dim, latent_dim))
        self.b2 = np.zeros((1, latent_dim))
        self.W3 = rng.normal(0.0, np.sqrt(
            2.0 / (latent_dim + hidden_dim)), size=(latent_dim, hidden_dim))
        self.b3 = np.zeros((1, hidden_dim))
        self.W4 = rng.normal(0.0, np.sqrt(
            2.0 / (hidden_dim + self.output_dim)), size=(hidden_dim, self.output_dim))
        self.b4 = np.zeros((1, self.output_dim))

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        z1 = features @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        latent = np.tanh(z2)
        z3 = latent @ self.W3 + self.b3
        a3 = np.tanh(z3)
        z4 = a3 @ self.W4 + self.b4
        output = self._sigmoid(z4)
        cache = {"a1": a1, "latent": latent, "a3": a3, "output": output}
        return output, latent, cache

    def reconstruction_loss(
        self,
        features: np.ndarray,
        target_features: np.ndarray | None = None,
    ) -> float:
        reconstructed, _, _ = self._forward(features)
        target = features if target_features is None else target_features
        return float(np.mean((reconstructed - target) ** 2))

    def fit(
        self,
        train_features: np.ndarray,
        target_train_features: np.ndarray | None = None,
        val_features: np.ndarray | None = None,
        target_val_features: np.ndarray | None = None,
        epochs: int = 2500,
        learning_rate: float = 0.03,
        batch_size: int = 32,
    ) -> dict[str, list[float]]:
        rng = np.random.default_rng(self.seed)
        history = {"train_loss": [], "val_loss": []}
        train_target = train_features if target_train_features is None else target_train_features
        val_target = val_features if target_val_features is None else target_val_features

        for epoch in range(epochs):
            order = rng.permutation(len(train_features))
            shuffled = train_features[order]
            shuffled_target = train_target[order]

            for start in range(0, len(shuffled), batch_size):
                batch = shuffled[start: start + batch_size]
                target_batch = shuffled_target[start: start + batch_size]
                reconstructed, _, cache = self._forward(batch)
                batch_size_actual = batch.shape[0]
                scale = 2.0 / (batch_size_actual * target_batch.shape[1])

                delta4 = scale * (reconstructed - target_batch) * \
                    reconstructed * (1.0 - reconstructed)
                dW4 = cache["a3"].T @ delta4
                db4 = np.sum(delta4, axis=0, keepdims=True)

                delta3 = (delta4 @ self.W4.T) * (1.0 - cache["a3"] ** 2)
                dW3 = cache["latent"].T @ delta3
                db3 = np.sum(delta3, axis=0, keepdims=True)

                delta2 = (delta3 @ self.W3.T) * (1.0 - cache["latent"] ** 2)
                dW2 = cache["a1"].T @ delta2
                db2 = np.sum(delta2, axis=0, keepdims=True)

                delta1 = (delta2 @ self.W2.T) * (1.0 - cache["a1"] ** 2)
                dW1 = batch.T @ delta1
                db1 = np.sum(delta1, axis=0, keepdims=True)

                self.W4 -= learning_rate * dW4
                self.b4 -= learning_rate * db4
                self.W3 -= learning_rate * dW3
                self.b3 -= learning_rate * db3
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1

            if epoch % 25 == 0 or epoch == epochs - 1:
                history["train_loss"].append(
                    self.reconstruction_loss(train_features, train_target))
                if val_features is not None:
                    history["val_loss"].append(
                        self.reconstruction_loss(val_features, val_target))

        return history

    def reconstruct(self, features: np.ndarray) -> np.ndarray:
        reconstructed, _, _ = self._forward(features)
        return reconstructed

    def encode(self, features: np.ndarray) -> np.ndarray:
        _, latent, _ = self._forward(features)
        return latent
