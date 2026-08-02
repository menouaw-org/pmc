import numpy as np
from tqdm import tqdm


class MLP:
    def __init__(self, npl: list[int], seed: int | None = 24) -> None:
        self.d = list(npl)
        random_generator = np.random.default_rng(seed)
        self.weights = [
            random_generator.uniform(
                -1.0,
                1.0,
                size=(previous_size + 1, current_size),
            )
            for previous_size, current_size in zip(self.d[:-1], self.d[1:], strict=True)
        ]

    @staticmethod
    def _add_bias(values: np.ndarray) -> np.ndarray:
        bias = np.ones((values.shape[0], 1))
        return np.concatenate((bias, values), axis=1)

    def _forward(self, inputs: np.ndarray) -> list[np.ndarray]:
        activations = [inputs]

        for weights in self.weights:
            previous_with_bias = self._add_bias(activations[-1])
            activations.append(np.tanh(previous_with_bias @ weights))

        return activations

    def predict(
        self,
        inputs: list[float] | list[list[float]] | np.ndarray,
    ) -> list[float] | np.ndarray:
        values = np.asarray(inputs, dtype=float)
        single_example = values.ndim == 1

        if single_example:
            values = values.reshape(1, -1)

        outputs = self._forward(values)[-1]
        return outputs[0].tolist() if single_example else outputs

    def train(
        self,
        dataset_inputs: list[list[float]] | np.ndarray,
        dataset_expected_outputs: list[list[float]] | np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int | None = 24,
    ) -> dict[str, list[float]]:
        inputs = np.asarray(dataset_inputs, dtype=float)
        expected_outputs = np.asarray(dataset_expected_outputs, dtype=float)
        random_generator = np.random.default_rng(seed)
        history: dict[str, list[float]] = {"loss": [], "accuracy": []}
        example_indices = np.arange(len(inputs))

        for _ in tqdm(range(epochs), desc="Entraînement"):
            if shuffle:
                random_generator.shuffle(example_indices)

            for batch_start in range(0, len(inputs), batch_size):
                batch_indices = example_indices[batch_start : batch_start + batch_size]
                batch_inputs = inputs[batch_indices]
                batch_targets = expected_outputs[batch_indices]
                activations = self._forward(batch_inputs)

                delta = (activations[-1] - batch_targets) * (1.0 - activations[-1] ** 2)

                for layer_index in reversed(range(len(self.weights))):
                    previous_activations = activations[layer_index]
                    previous_with_bias = self._add_bias(previous_activations)
                    gradient = previous_with_bias.T @ delta / len(batch_indices)

                    if layer_index == 0:
                        self.weights[layer_index] -= learning_rate * gradient
                    else:
                        propagated_delta = delta @ self.weights[layer_index][1:, :].T
                        next_delta = propagated_delta * (1.0 - previous_activations**2)
                        self.weights[layer_index] -= learning_rate * gradient
                        delta = next_delta

            predictions = self._forward(inputs)[-1]
            loss = np.mean((predictions - expected_outputs) ** 2) / 2.0
            history["loss"].append(float(loss))
            history["accuracy"].append(self._accuracy(predictions, expected_outputs))

        return history

    @staticmethod
    def _accuracy(predictions: np.ndarray, expected_outputs: np.ndarray) -> float:
        if predictions.shape[1] == 1:
            predicted_classes = np.where(predictions[:, 0] >= 0.0, 1, -1)
            expected_classes = np.where(expected_outputs[:, 0] >= 0.0, 1, -1)
        else:
            predicted_classes = np.argmax(predictions, axis=1)
            expected_classes = np.argmax(expected_outputs, axis=1)

        return float(np.mean(predicted_classes == expected_classes))

    def save(self, file_path: str) -> None:
        values = {"architecture": np.asarray(self.d)}
        values.update(
            {f"weights_{index}": weights for index, weights in enumerate(self.weights)}
        )
        np.savez(file_path, **values)

    @classmethod
    def load(cls, file_path: str) -> "MLP":
        with np.load(file_path) as saved_values:
            architecture = saved_values["architecture"].tolist()
            model = cls(architecture, seed=0)
            model.weights = [
                saved_values[f"weights_{index}"].copy()
                for index in range(len(architecture) - 1)
            ]

        return model