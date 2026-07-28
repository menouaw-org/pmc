import random
import shutil
from collections import Counter
from pathlib import Path

PROCESSED_ROOT = Path("data/processed")
CLASSES = ("avion", "montgolfiere", "parapente")
SIZES = (128, 64, 32)
COLOR_MODES = ("rgb", "grayscale")
SPLITS = ("train", "validation", "test")
RANDOM_SEED = 24
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.10
REFERENCE_SIZE = 128
REFERENCE_MODE = "rgb"


def list_jpg(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Dossier introuvable: {directory}")

    files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if not files:
        raise RuntimeError(f"Aucune image JPG trouvée dans {directory}")

    return files


def variant_root(size: int, color_mode: str) -> Path:
    return PROCESSED_ROOT / f"{size}x{size}" / color_mode


def reference_names_by_class() -> dict[str, set[str]]:
    return {
        class_name: {
            path.name
            for path in list_jpg(variant_root(REFERENCE_SIZE, REFERENCE_MODE) / class_name)
        }
        for class_name in CLASSES
    }


def validate_source_variants(reference_names: dict[str, set[str]]) -> None:
    for size in SIZES:
        for color_mode in COLOR_MODES:
            root = variant_root(size, color_mode)
            for class_name in CLASSES:
                actual_names = {path.name for path in list_jpg(root / class_name)}
                expected_names = reference_names[class_name]

                if actual_names != expected_names:
                    missing = sorted(expected_names - actual_names)
                    unexpected = sorted(actual_names - expected_names)
                    raise RuntimeError(
                        f"Variante incohérente dans {root / class_name}. "
                        f"Manquants: {missing[:5]}; inattendus: {unexpected[:5]}"
                    )


def ensure_split_directories_absent() -> None:
    for size in SIZES:
        for color_mode in COLOR_MODES:
            root = variant_root(size, color_mode)
            for split_name in SPLITS:
                split_dir = root / split_name
                if split_dir.exists():
                    raise FileExistsError(
                        f"Dossier de séparation déjà présent: {split_dir}. "
                        "Reconstruisez data/processed avant de relancer le script."
                    )


def build_assignments(
    reference_names: dict[str, set[str]],
) -> dict[str, dict[str, str]]:
    random_generator = random.Random(RANDOM_SEED)
    assignments: dict[str, dict[str, str]] = {}

    for class_name in CLASSES:
        names = sorted(reference_names[class_name])
        random_generator.shuffle(names)

        validation_count = round(len(names) * VALIDATION_RATIO)
        test_count = round(len(names) * TEST_RATIO)
        train_count = len(names) - validation_count - test_count

        class_assignments: dict[str, str] = {}
        for name in names[:train_count]:
            class_assignments[name] = "train"
        for name in names[train_count : train_count + validation_count]:
            class_assignments[name] = "validation"
        for name in names[train_count + validation_count :]:
            class_assignments[name] = "test"

        assignments[class_name] = class_assignments

    return assignments


def move_files(assignments: dict[str, dict[str, str]]) -> None:
    for size in SIZES:
        for color_mode in COLOR_MODES:
            root = variant_root(size, color_mode)

            for class_name in CLASSES:
                source_dir = root / class_name
                source_files = list_jpg(source_dir)

                for source_path in source_files:
                    split_name = assignments[class_name][source_path.name]
                    destination = root / split_name / class_name / source_path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(source_path, destination)

                source_dir.rmdir()


def validate_outputs(assignments: dict[str, dict[str, str]]) -> None:
    for size in SIZES:
        for color_mode in COLOR_MODES:
            root = variant_root(size, color_mode)

            for split_name in SPLITS:
                for class_name in CLASSES:
                    expected_names = {
                        name
                        for name, assigned_split in assignments[class_name].items()
                        if assigned_split == split_name
                    }
                    output_dir = root / split_name / class_name
                    actual_names = {path.name for path in list_jpg(output_dir)}

                    if actual_names != expected_names:
                        raise RuntimeError(f"Séparation incohérente dans {output_dir}")


def print_summary(assignments: dict[str, dict[str, str]]) -> None:
    totals = Counter()

    for class_name in CLASSES:
        class_counts = Counter(assignments[class_name].values())
        totals.update(class_counts)
        print(
            f"{class_name}: "
            f"train={class_counts['train']}, "
            f"validation={class_counts['validation']}, "
            f"test={class_counts['test']}"
        )

    print(
        "Total par variante: "
        f"train={totals['train']}, "
        f"validation={totals['validation']}, "
        f"test={totals['test']}"
    )
    print(f"Fichiers physiques conservés: {sum(totals.values()) * 6}")


def main() -> None:
    reference_names = reference_names_by_class()
    validate_source_variants(reference_names)
    ensure_split_directories_absent()

    assignments = build_assignments(reference_names)
    move_files(assignments)
    validate_outputs(assignments)
    print_summary(assignments)


if __name__ == "__main__":
    main()
