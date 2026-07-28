from pathlib import Path

from PIL import Image, ImageOps

RAW_ROOT = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")
CLASSES = ("avion", "montgolfiere", "parapente")
SIZES = (128, 64, 32)
COLOR_MODES = {"rgb": "RGB", "grayscale": "L"}
JPEG_QUALITY = 95


def list_raw_images(class_name: str) -> list[Path]:
    class_dir = RAW_ROOT / class_name
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Dossier brut introuvable: {class_dir}")

    files = sorted(
        path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"
    )
    if not files:
        raise RuntimeError(f"Aucune image JPG trouvée dans {class_dir}")

    return files


def crop_and_resize(source_path: Path, size: int) -> Image.Image:
    with Image.open(source_path) as source:
        oriented = ImageOps.exif_transpose(source)
        rgb_image = oriented.convert("RGB")
        return ImageOps.fit(
            rgb_image,
            (size, size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


def save_variant(
    image: Image.Image,
    output_path: Path,
    target_mode: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variant = image if target_mode == "RGB" else image.convert(target_mode)
    variant.save(
        output_path,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=True,
    )


def validate_variant(
    raw_files: list[Path],
    class_name: str,
    size: int,
    variant_name: str,
    target_mode: str,
) -> None:
    output_dir = PROCESSED_ROOT / f"{size}x{size}" / variant_name / class_name
    expected_names = {path.name for path in raw_files}
    output_files = sorted(output_dir.glob("*.jpg"))
    actual_names = {path.name for path in output_files}

    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise RuntimeError(
            f"Contenu incohérent dans {output_dir}. "
            f"Manquants: {missing[:5]}; inattendus: {unexpected[:5]}"
        )

    for output_path in output_files:
        with Image.open(output_path) as image:
            if image.size != (size, size):
                raise RuntimeError(f"Dimensions incorrectes pour {output_path}: {image.size}")
            if image.mode != target_mode:
                raise RuntimeError(f"Mode incorrect pour {output_path}: {image.mode}")


def main() -> None:
    raw_by_class = {class_name: list_raw_images(class_name) for class_name in CLASSES}
    total_raw = sum(len(files) for files in raw_by_class.values())

    for class_name, raw_files in raw_by_class.items():
        for source_path in raw_files:
            for size in SIZES:
                resized = crop_and_resize(source_path, size)

                for variant_name, target_mode in COLOR_MODES.items():
                    output_path = (
                        PROCESSED_ROOT
                        / f"{size}x{size}"
                        / variant_name
                        / class_name
                        / source_path.name
                    )
                    save_variant(resized, output_path, target_mode)

    for class_name, raw_files in raw_by_class.items():
        for size in SIZES:
            for variant_name, target_mode in COLOR_MODES.items():
                validate_variant(
                    raw_files,
                    class_name,
                    size,
                    variant_name,
                    target_mode,
                )

    generated_count = total_raw * len(SIZES) * len(COLOR_MODES)
    print(f"Images brutes: {total_raw}")
    print(f"Fichiers traités et validés: {generated_count}")


if __name__ == "__main__":
    main()
