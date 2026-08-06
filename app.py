from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from pmc import MLP

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "final_32x32_grayscale_seed_24" / "model.npz"
RESOLUTION = 32
CLASS_LABELS = ("avion", "montgolfière", "parapente")


def load_model() -> MLP:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Artefact du modèle introuvable: {MODEL_PATH}")

    model = MLP.load(str(MODEL_PATH))
    expected_input_size = RESOLUTION**2

    if model.d[0] != expected_input_size or model.d[-1] != len(CLASS_LABELS):
        raise ValueError(
            "Architecture de l’artefact incompatible: "
            f"entrée={model.d[0]}, sortie={model.d[-1]}, "
            f"attendu=({expected_input_size}, {len(CLASS_LABELS)})."
        )

    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    oriented_image = ImageOps.exif_transpose(image)
    rgb_image = oriented_image.convert("RGB")
    resized_image = ImageOps.fit(
        rgb_image,
        (RESOLUTION, RESOLUTION),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    grayscale_image = resized_image.convert("L")
    pixels = np.asarray(grayscale_image, dtype=float)
    return pixels.reshape(-1) / 127.5 - 1.0


def normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    shifted_scores = raw_scores - np.max(raw_scores)
    exponentials = np.exp(shifted_scores)
    return exponentials / np.sum(exponentials)


MODEL = load_model()


def predict_image(image: Image.Image | None) -> dict[str, float]:
    if image is None:
        raise ValueError("Aucune image n’a été fournie.")

    inputs = preprocess_image(image)
    raw_scores = np.asarray(MODEL.predict(inputs), dtype=float)

    if raw_scores.shape != (len(CLASS_LABELS),):
        raise RuntimeError(
            f"Dimension de sortie inattendue: {raw_scores.shape}, "
            f"attendu={(len(CLASS_LABELS),)}."
        )

    display_scores = normalize_scores(raw_scores)
    return {
        label: float(score)
        for label, score in zip(CLASS_LABELS, display_scores, strict=True)
    }


def build_interface() -> gr.Interface:
    return gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Image à classer"),
        outputs=gr.Label(
            num_top_classes=len(CLASS_LABELS),
            label="Prédiction",
        ),
        title="Classification d’images par PMC",
        description=(
            "Déposez une image d’avion, de montgolfière ou de parapente. "
            "Le modèle pré-entraîné est exécuté localement."
        ),
    )


demo = build_interface()


if __name__ == "__main__":
    demo.launch()