import math
import random

from tqdm import tqdm


class NaiveMLP:
    def __init__(self, npl: list[int]) -> None:
        self.d = list(npl)
        self.L = len(npl) - 1

        self.W: list[list[list[float]]] = []
        for layer in range(self.L + 1):
            self.W.append([])

            if layer == 0:
                continue

            for previous_neuron in range(self.d[layer - 1] + 1):
                self.W[layer].append([])

                for current_neuron in range(self.d[layer] + 1):
                    if current_neuron == 0:
                        weight = 0.0
                    else:
                        weight = random.random() * 2.0 - 1.0

                    self.W[layer][previous_neuron].append(weight)

        self.X: list[list[float]] = []
        self.deltas: list[list[float]] = []

        for layer in range(self.L + 1):
            self.X.append([])
            self.deltas.append([])

            for neuron in range(self.d[layer] + 1):
                self.X[layer].append(1.0 if neuron == 0 else 0.0)
                self.deltas[layer].append(0.0)

    def _propagate(self, inputs: list[float]) -> None:
        for neuron in range(1, self.d[0] + 1):
            self.X[0][neuron] = inputs[neuron - 1]

        for layer in range(1, self.L + 1):
            for current_neuron in range(1, self.d[layer] + 1):
                total = 0.0

                for previous_neuron in range(self.d[layer - 1] + 1):
                    total += (
                        self.W[layer][previous_neuron][current_neuron]
                        * self.X[layer - 1][previous_neuron]
                    )

                self.X[layer][current_neuron] = math.tanh(total)

    def predict(self, inputs: list[float]) -> list[float]:
        self._propagate(inputs)
        return self.X[self.L][1:]

    def train(
        self,
        dataset_inputs: list[list[float]],
        dataset_expected_outputs: list[list[float]],
        training_steps: int,
        learning_rate: float,
    ) -> None:
        for _ in tqdm(range(training_steps)):
            example_index = random.randint(0, len(dataset_inputs) - 1)
            inputs = dataset_inputs[example_index]
            expected_outputs = dataset_expected_outputs[example_index]

            self._propagate(inputs)

            for neuron in range(1, self.d[self.L] + 1):
                output = self.X[self.L][neuron]
                error = output - expected_outputs[neuron - 1]
                self.deltas[self.L][neuron] = error * (1.0 - output**2)

            for layer in reversed(range(2, self.L + 1)):
                for previous_neuron in range(1, self.d[layer - 1] + 1):
                    total = 0.0

                    for current_neuron in range(1, self.d[layer] + 1):
                        total += (
                            self.W[layer][previous_neuron][current_neuron]
                            * self.deltas[layer][current_neuron]
                        )

                    previous_output = self.X[layer - 1][previous_neuron]
                    self.deltas[layer - 1][previous_neuron] = total * (
                        1.0 - previous_output**2
                    )

            for layer in range(1, self.L + 1):
                for previous_neuron in range(self.d[layer - 1] + 1):
                    for current_neuron in range(1, self.d[layer] + 1):
                        self.W[layer][previous_neuron][current_neuron] -= (
                            learning_rate
                            * self.X[layer - 1][previous_neuron]
                            * self.deltas[layer][current_neuron]
                        )