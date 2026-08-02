from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = Path("data/processed/32x32/grayscale")
CLASS_NAMES = ("avion", "montgolfiere", "parapente")


def bipolar_target(class_index: int) -> np.ndarray:
    target = np.full(len(CLASS_NAMES), -1.0)
    target[class_index] = 1.0
    return target


def load_split(split_name: str) -> tuple[np.ndarray, np.ndarray]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_directory = DATA_ROOT / split_name / class_name

        for image_path in sorted(class_directory.glob("*.jpg")):
            with Image.open(image_path) as image:
                pixels = np.asarray(image, dtype=float) / 127.5 - 1.0

            inputs.append(pixels.reshape(-1))
            targets.append(bipolar_target(class_index))

    return np.asarray(inputs), np.asarray(targets)