from pathlib import Path
from typing import List

from .pdf_processor import pdf_to_images


SUPPORTED_INPUT_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}


def load_document_images(input_path: Path) -> List:
    """Load a PDF or image input into a list of RGB PIL images."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pdf_to_images(path)

    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
        raise ValueError(f"Unsupported input file type '{suffix}'. Allowed: {allowed}")

    try:
        from PIL import Image, ImageOps, ImageSequence
    except Exception as exc:
        raise ImportError("Pillow is required to read image uploads.") from exc

    pages = []
    with Image.open(path) as image:
        for frame in ImageSequence.Iterator(image):
            normalized = ImageOps.exif_transpose(frame).copy()
            pages.append(_ensure_rgb(normalized))

    if not pages:
        raise ValueError(f"No readable pages found in {path}")

    return pages


def is_supported_input_file(path: Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_INPUT_SUFFIXES


def _ensure_rgb(image):
    if image.mode == "RGB":
        return image

    if "A" in image.getbands():
        from PIL import Image

        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background

    return image.convert("RGB")
