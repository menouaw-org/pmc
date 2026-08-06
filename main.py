import math

from pmc import MLP


def run_classification(
    name: str,
    architecture: list[int],
    dataset_inputs: list[list[float]],
    dataset_expected_outputs: list[list[float]],
    training_steps: int,
) -> None:
    model = MLP(architecture, seed=24)
    model.train(
        dataset_inputs,
        dataset_expected_outputs,
        epochs=math.ceil(training_steps / len(dataset_inputs)),
        learning_rate=0.01,
    )

    print(f"\n{name}")
    for inputs, expected_outputs in zip(
        dataset_inputs,
        dataset_expected_outputs,
        strict=True,
    ):
        predicted_output = model.predict(inputs)[0]
        expected_output = expected_outputs[0]
        predicted_class = 1.0 if predicted_output >= 0.0 else -1.0
        print(
            f"entrées={inputs} | "
            f"cible={expected_output:+.0f} | "
            f"sortie={predicted_output:+.4f} | "
            f"classe={predicted_class:+.0f}"
        )


def main() -> None:
    run_classification(
        name="Classification simple — AND",
        architecture=[2, 1],
        dataset_inputs=[
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dataset_expected_outputs=[
            [-1.0],
            [-1.0],
            [-1.0],
            [1.0],
        ],
        training_steps=10_000,
    )

    run_classification(
        name="Classification non linéaire — XOR",
        architecture=[2, 3, 1],
        dataset_inputs=[
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dataset_expected_outputs=[
            [-1.0],
            [1.0],
            [1.0],
            [-1.0],
        ],
        training_steps=100_000,
    )


if __name__ == "__main__":
    main()