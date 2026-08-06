from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import wandb

from pmc.image_dataset import CLASS_NAMES, load_split
from pmc.mlp import MLP

ARCHITECTURE = [1024, 32, 3]
EPOCHS = 50
LEARNING_RATE = 0.01
BATCH_SIZE = 32
SEED = 24
ARTIFACT_DIRECTORY = Path("artifacts/baseline_32x32_grayscale")
MODEL_PATH = ARTIFACT_DIRECTORY / "model.npz"
CURVES_PATH = ARTIFACT_DIRECTORY / "training_curves.png"
CONFUSION_MATRIX_PATH = ARTIFACT_DIRECTORY / "confusion_matrix.png"


def confusion_matrix(expected_classes: np.ndarray, predicted_classes: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)

    for expected_class, predicted_class in zip(
        expected_classes,
        predicted_classes,
        strict=True,
    ):
        matrix[expected_class, predicted_class] += 1

    return matrix


def save_training_curves(history: dict[str, list[float]]) -> None:
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
    figure.savefig(CURVES_PATH, dpi=150)
    plt.close(figure)


def save_confusion_matrix(matrix: np.ndarray) -> None:
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
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(figure)


def main() -> None:
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    train_inputs, train_targets = load_split("train")
    validation_inputs, validation_targets = load_split("validation")

    model = MLP(ARCHITECTURE, seed=SEED)

    with wandb.init(
        project="pmc",
        name="baseline-32x32-grayscale",
        job_type="baseline",
        config={
            "architecture": ARCHITECTURE,
            "resolution": 32,
            "color_mode": "grayscale",
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
    ) as run:
        start_time = perf_counter()
        history = model.train(
            train_inputs,
            train_targets,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=SEED,
            validation_inputs=validation_inputs,
            validation_expected_outputs=validation_targets,
        )
        duration_seconds = perf_counter() - start_time

        for epoch in range(EPOCHS):
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
        matrix = confusion_matrix(expected_classes, predicted_classes)

        model.save(str(MODEL_PATH))
        save_training_curves(history)
        save_confusion_matrix(matrix)

        reloaded_model = MLP.load(str(MODEL_PATH))
        reloaded_outputs = np.asarray(reloaded_model.predict(validation_inputs))
        maximum_reload_difference = float(
            np.max(np.abs(validation_outputs - reloaded_outputs))
        )

        run.summary["duration_seconds"] = duration_seconds
        run.summary["final_train_loss"] = history["loss"][-1]
        run.summary["final_train_accuracy"] = history["accuracy"][-1]
        run.summary["final_validation_loss"] = history["validation_loss"][-1]
        run.summary["final_validation_accuracy"] = history["validation_accuracy"][-1]
        run.summary["maximum_reload_difference"] = maximum_reload_difference
        run.log(
            {
                "training_curves": wandb.Image(str(CURVES_PATH)),
                "confusion_matrix": wandb.Image(str(CONFUSION_MATRIX_PATH)),
            }
        )

        model_artifact = wandb.Artifact("baseline-32x32-grayscale", type="model")
        model_artifact.add_file(str(MODEL_PATH))
        run.log_artifact(model_artifact)

        print(f"durée={duration_seconds:.2f}s")
        print(f"train_loss={history['loss'][-1]:.6f}")
        print(f"train_accuracy={history['accuracy'][-1]:.4f}")
        print(f"validation_loss={history['validation_loss'][-1]:.6f}")
        print(f"validation_accuracy={history['validation_accuracy'][-1]:.4f}")
        print(f"maximum_reload_difference={maximum_reload_difference:.12f}")
        print("confusion_matrix=")
        print(matrix)


if __name__ == "__main__":
    main()