import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import wandb

from pmc.image_dataset import CLASS_NAMES, load_split
from pmc.mlp import MLP

RESOLUTION = 32
COLOR_MODE = "grayscale"
ARCHITECTURE = [1024, 32, 3]
EPOCHS = 200
LEARNING_RATE = 0.01
BATCH_SIZE = 32
SEED = 24
RUN_NAME = "final-32x32-grayscale-seed-24"
ARTIFACT_DIRECTORY = Path("artifacts/final_32x32_grayscale_seed_24")
MODEL_PATH = ARTIFACT_DIRECTORY / "model.npz"
CURVES_PATH = ARTIFACT_DIRECTORY / "training_curves.png"
CONFUSION_MATRIX_PATH = ARTIFACT_DIRECTORY / "test_confusion_matrix.png"
METRICS_PATH = ARTIFACT_DIRECTORY / "metrics.json"


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


def save_training_curves(history: dict[str, list[float]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], label="train + validation")
    axes[0].set_title("Perte d'entraînement final")
    axes[0].set_xlabel("Époque")
    axes[0].legend()
    axes[1].plot(history["accuracy"], label="train + validation")
    axes[1].set_title("Exactitude d'entraînement final")
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
    axis.set_title("Matrice de confusion — test final")

    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")

    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(figure)


def main() -> None:
    if ARTIFACT_DIRECTORY.exists():
        raise FileExistsError(
            f"Évaluation finale déjà préparée dans {ARTIFACT_DIRECTORY}. "
            "Ne supprimez pas ce dossier pour relancer le test avec une autre configuration."
        )

    ARTIFACT_DIRECTORY.mkdir(parents=True)
    train_inputs, train_targets = load_split(
        "train",
        resolution=RESOLUTION,
        color_mode=COLOR_MODE,
    )
    validation_inputs, validation_targets = load_split(
        "validation",
        resolution=RESOLUTION,
        color_mode=COLOR_MODE,
    )
    final_inputs = np.concatenate((train_inputs, validation_inputs), axis=0)
    final_targets = np.concatenate((train_targets, validation_targets), axis=0)

    if len(final_inputs) != 1164:
        raise RuntimeError(f"Effectif train + validation inattendu: {len(final_inputs)}")

    model = MLP(ARCHITECTURE, seed=SEED)

    with wandb.init(
        project="pmc",
        name=RUN_NAME,
        job_type="final-evaluation",
        config={
            "architecture": ARCHITECTURE,
            "resolution": RESOLUTION,
            "color_mode": COLOR_MODE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "training_split": "train+validation",
            "training_examples": 1164,
            "test_examples": 129,
        },
    ) as run:
        start_time = perf_counter()
        history = model.train(
            final_inputs,
            final_targets,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=SEED,
        )
        duration_seconds = perf_counter() - start_time

        for epoch in range(EPOCHS):
            run.log(
                {
                    "epoch": epoch + 1,
                    "final_train/loss": history["loss"][epoch],
                    "final_train/accuracy": history["accuracy"][epoch],
                },
                step=epoch + 1,
            )

        test_inputs, test_targets = load_split(
            "test",
            resolution=RESOLUTION,
            color_mode=COLOR_MODE,
        )
        if len(test_inputs) != 129:
            raise RuntimeError(f"Effectif de test inattendu: {len(test_inputs)}")

        test_outputs = np.asarray(model.predict(test_inputs))
        predicted_classes = np.argmax(test_outputs, axis=1)
        expected_classes = np.argmax(test_targets, axis=1)
        matrix = build_confusion_matrix(expected_classes, predicted_classes)
        test_loss = float(np.mean((test_outputs - test_targets) ** 2) / 2.0)
        test_accuracy = float(np.mean(predicted_classes == expected_classes))
        class_totals = matrix.sum(axis=1)
        class_recalls = matrix.diagonal() / class_totals

        model.save(str(MODEL_PATH))
        save_training_curves(history)
        save_confusion_matrix(matrix)

        reloaded_model = MLP.load(str(MODEL_PATH))
        reloaded_outputs = np.asarray(reloaded_model.predict(test_inputs))
        maximum_reload_difference = float(np.max(np.abs(test_outputs - reloaded_outputs)))
        metrics = {
            "architecture": ARCHITECTURE,
            "resolution": RESOLUTION,
            "color_mode": COLOR_MODE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "training_examples": len(final_inputs),
            "test_examples": len(test_inputs),
            "duration_seconds": duration_seconds,
            "final_train_loss": history["loss"][-1],
            "final_train_accuracy": history["accuracy"][-1],
            "final_test_loss": test_loss,
            "final_test_accuracy": test_accuracy,
            "test_recall_avion": float(class_recalls[0]),
            "test_recall_montgolfiere": float(class_recalls[1]),
            "test_recall_parapente": float(class_recalls[2]),
            "maximum_reload_difference": maximum_reload_difference,
            "confusion_matrix": matrix.tolist(),
            "wandb_run_url": run.url,
        }
        METRICS_PATH.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        run.summary.update(metrics)
        run.log(
            {
                "training_curves": wandb.Image(str(CURVES_PATH)),
                "test_confusion_matrix": wandb.Image(str(CONFUSION_MATRIX_PATH)),
            }
        )
        artifact = wandb.Artifact(RUN_NAME, type="model")
        artifact.add_file(str(MODEL_PATH))
        artifact.add_file(str(METRICS_PATH))
        artifact.add_file(str(CURVES_PATH))
        artifact.add_file(str(CONFUSION_MATRIX_PATH))
        run.log_artifact(artifact)

        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
