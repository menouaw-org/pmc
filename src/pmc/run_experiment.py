import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

import wandb
from pmc.image_dataset import CLASS_NAMES, load_split
from pmc.mlp import MLP

ARTIFACT_ROOT = Path("artifacts/experiments")
DEFAULT_SEEDS = (7, 24, 42, 123, 2026)


def parse_hidden_layers(raw_value: str) -> list[int]:
    return [int(value) for value in raw_value.split(",")]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resolution", type=int, choices=(32, 64, 128), required=True)
    parser.add_argument("--color-mode", choices=("grayscale", "rgb"), required=True)
    parser.add_argument("--hidden-layers", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seeds", type=int, nargs="+")
    return parser.parse_args()


def build_confusion_matrix(
    expected_classes: np.ndarray,
    predicted_classes: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)

    for expected_class, predicted_class in zip(
        expected_classes,
        predicted_classes,
        strict=True,
    ):
        matrix[expected_class, predicted_class] += 1

    return matrix


def save_training_curves(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["loss"], label="apprentissage")
    axes[0].plot(history["validation_loss"], label="validation")
    axes[0].set_title("Perte")
    axes[0].set_xlabel("Époque")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="apprentissage")
    axes[1].plot(history["validation_accuracy"], label="validation")
    axes[1].set_title("Exactitude")
    axes[1].set_xlabel("Époque")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Classe prédite")
    axis.set_ylabel("Classe attendue")
    axis.set_title("Matrice de confusion — validation")

    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_experiment(
    arguments: argparse.Namespace,
    seed: int,
    run_name: str,
) -> None:
    hidden_layers = parse_hidden_layers(arguments.hidden_layers)
    channel_count = 1 if arguments.color_mode == "grayscale" else 3
    input_dimension = arguments.resolution**2 * channel_count
    architecture = [input_dimension, *hidden_layers, len(CLASS_NAMES)]

    artifact_directory = ARTIFACT_ROOT / run_name
    artifact_directory.mkdir(parents=True, exist_ok=True)
    model_path = artifact_directory / "model.npz"
    curves_path = artifact_directory / "training_curves.png"
    confusion_matrix_path = artifact_directory / "confusion_matrix.png"
    metrics_path = artifact_directory / "metrics.json"

    train_inputs, train_targets = load_split(
        "train",
        resolution=arguments.resolution,
        color_mode=arguments.color_mode,
    )
    validation_inputs, validation_targets = load_split(
        "validation",
        resolution=arguments.resolution,
        color_mode=arguments.color_mode,
    )
    model = MLP(architecture, seed=seed)

    with wandb.init(
        project="pmc",
        name=run_name,
        job_type="experiment",
        config={
            "architecture": architecture,
            "resolution": arguments.resolution,
            "color_mode": arguments.color_mode,
            "epochs": arguments.epochs,
            "learning_rate": arguments.learning_rate,
            "batch_size": arguments.batch_size,
            "seed": seed,
        },
    ) as run:
        start_time = perf_counter()
        history = model.train(
            train_inputs,
            train_targets,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            batch_size=arguments.batch_size,
            shuffle=True,
            seed=seed,
            validation_inputs=validation_inputs,
            validation_expected_outputs=validation_targets,
        )
        duration_seconds = perf_counter() - start_time

        for epoch in range(arguments.epochs):
            run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": history["loss"][epoch],
                    "train/accuracy": history["accuracy"][epoch],
                    "validation/loss": history["validation_loss"][epoch],
                    "validation/accuracy": history["validation_accuracy"][epoch],
                },
                step=epoch + 1,
            )

        validation_outputs = np.asarray(model.predict(validation_inputs))
        predicted_classes = np.argmax(validation_outputs, axis=1)
        expected_classes = np.argmax(validation_targets, axis=1)
        matrix = build_confusion_matrix(expected_classes, predicted_classes)

        model.save(str(model_path))
        save_training_curves(history, curves_path)
        save_confusion_matrix(matrix, confusion_matrix_path)

        reloaded_model = MLP.load(str(model_path))
        reloaded_outputs = np.asarray(reloaded_model.predict(validation_inputs))
        maximum_reload_difference = float(
            np.max(np.abs(validation_outputs - reloaded_outputs))
        )
        metrics = {
            "architecture": architecture,
            "resolution": arguments.resolution,
            "color_mode": arguments.color_mode,
            "epochs": arguments.epochs,
            "learning_rate": arguments.learning_rate,
            "batch_size": arguments.batch_size,
            "seed": seed,
            "duration_seconds": duration_seconds,
            "final_train_loss": history["loss"][-1],
            "final_train_accuracy": history["accuracy"][-1],
            "final_validation_loss": history["validation_loss"][-1],
            "final_validation_accuracy": history["validation_accuracy"][-1],
            "maximum_reload_difference": maximum_reload_difference,
            "confusion_matrix": matrix.tolist(),
            "wandb_run_url": run.url,
        }
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        run.summary.update(metrics)
        run.log(
            {
                "training_curves": wandb.Image(str(curves_path)),
                "confusion_matrix": wandb.Image(str(confusion_matrix_path)),
            }
        )
        artifact = wandb.Artifact(run_name, type="model")
        artifact.add_file(str(model_path))
        artifact.add_file(str(metrics_path))
        run.log_artifact(artifact)

        print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main() -> None:
    arguments = parse_arguments()

    if arguments.seeds is not None:
        seeds = tuple(arguments.seeds)
    elif arguments.seed is not None:
        seeds = (arguments.seed,)
    else:
        seeds = DEFAULT_SEEDS

    for seed in seeds:
        run_name = (
            arguments.run_name
            if len(seeds) == 1
            else f"{arguments.run_name}-seed-{seed}"
        )
        run_experiment(arguments, seed, run_name)


if __name__ == "__main__":
    main()