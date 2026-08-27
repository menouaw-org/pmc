import numpy as np
from tqdm import tqdm


class MLP:
    def __init__(self, npl: list[int], seed: int | None = 24) -> None:
        self.d = list(npl)
        random_generator = np.random.default_rng(seed)
        self.weights = self._initialize_weights(random_generator)

    def _initialize_weights(self, random_generator: np.random.Generator) -> list[np.ndarray]:
        return [
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

    @staticmethod
    def _as_batch(inputs: list[float] | list[list[float]] | np.ndarray) -> tuple[np.ndarray, bool]:
        values = np.asarray(inputs, dtype=float)
        is_single_example = values.ndim == 1
        if is_single_example:
            values = values.reshape(1, -1)

        return values, is_single_example

    def predict(
        self,
        inputs: list[float] | list[list[float]] | np.ndarray,
    ) -> list[float] | np.ndarray:
        values, is_single_example = self._as_batch(inputs)
        outputs = self._forward(values)[-1]
        return outputs[0].tolist() if is_single_example else outputs

    def _tanh_derivative(self, activations: np.ndarray) -> np.ndarray:
        return 1.0 - activations**2

    def _output_delta(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return (predictions - targets) * self._tanh_derivative(predictions)

    def _previous_layer_delta(
        self,
        delta: np.ndarray,
        weights: np.ndarray,
        previous_activations: np.ndarray,
    ) -> np.ndarray:
        weights_without_bias = weights[1:, :]
        propagated_delta = delta @ weights_without_bias.T
        return propagated_delta * self._tanh_derivative(previous_activations)

    def _gradient(self, previous_activations: np.ndarray, delta: np.ndarray) -> np.ndarray:
        previous_with_bias = self._add_bias(previous_activations)
        return previous_with_bias.T @ delta / len(delta)

    def _train_batch(
        self,
        batch_inputs: np.ndarray,
        batch_targets: np.ndarray,
        learning_rate: float,
    ) -> None:
        activations = self._forward(batch_inputs)
        delta = self._output_delta(activations[-1], batch_targets)

        for layer_index in reversed(range(len(self.weights))):
            previous_activations = activations[layer_index]
            gradient = self._gradient(previous_activations, delta)

            previous_delta = None
            if layer_index > 0:
                previous_delta = self._previous_layer_delta(
                    delta,
                    self.weights[layer_index],
                    previous_activations,
                )

            self.weights[layer_index] -= learning_rate * gradient
            if previous_delta is not None:
                delta = previous_delta

    def train(
        self,
        dataset_inputs: list[list[float]] | np.ndarray,
        dataset_expected_outputs: list[list[float]] | np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int | None = 24,
        validation_inputs: list[list[float]] | np.ndarray | None = None,
        validation_expected_outputs: list[list[float]] | np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        inputs = np.asarray(dataset_inputs, dtype=float)
        expected_outputs = np.asarray(dataset_expected_outputs, dtype=float)
        random_generator = np.random.default_rng(seed)
        history: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
        }
        example_indices = np.arange(len(inputs))

        validation_values = None
        validation_targets = None
        if validation_inputs is not None and validation_expected_outputs is not None:
            validation_values = np.asarray(validation_inputs, dtype=float)
            validation_targets = np.asarray(validation_expected_outputs, dtype=float)

        for _ in tqdm(range(epochs), desc="Entraînement"):
            if shuffle:
                random_generator.shuffle(example_indices)

            for batch_start in range(0, len(inputs), batch_size):
                batch_indices = example_indices[batch_start : batch_start + batch_size]
                batch_inputs = inputs[batch_indices]
                batch_targets = expected_outputs[batch_indices]
                self._train_batch(batch_inputs, batch_targets, learning_rate)

            self._record_metrics(history, "loss", "accuracy", inputs, expected_outputs)

            if validation_values is not None and validation_targets is not None:
                self._record_metrics(
                    history,
                    "validation_loss",
                    "validation_accuracy",
                    validation_values,
                    validation_targets,
                )

        return history

    def _record_metrics(
        self,
        history: dict[str, list[float]],
        loss_key: str,
        accuracy_key: str,
        inputs: np.ndarray,
        expected_outputs: np.ndarray,
    ) -> None:
        predictions = self._forward(inputs)[-1]
        history[loss_key].append(self._loss(predictions, expected_outputs))
        history[accuracy_key].append(self._accuracy(predictions, expected_outputs))


    def _loss(self, predictions: np.ndarray, expected_outputs: np.ndarray) -> float:
        squared_errors = (predictions - expected_outputs) ** 2
        return float(np.mean(squared_errors) / 2.0)

    @staticmethod
    def _accuracy(predictions: np.ndarray, expected_outputs: np.ndarray) -> float:
        if predictions.shape[1] == 1:
            predicted_classes = np.where(
                predictions[:, 0] >= 0.0,
                1,
                -1,
            )
            expected_classes = np.where(
                expected_outputs[:, 0] >= 0.0,
                1,
                -1,
            )
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