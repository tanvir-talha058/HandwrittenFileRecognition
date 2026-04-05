from pathlib import Path
from typing import List


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> List:
    """Convert a PDF into per-page RGB images (PIL.Image objects)."""
    try:
        from pdf2image import convert_from_path
    except Exception as exc:
        raise ImportError(
            "pdf2image is required for PDF conversion. Install requirements first."
        ) from exc

    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    return convert_from_path(str(pdf_path), dpi=dpi)
