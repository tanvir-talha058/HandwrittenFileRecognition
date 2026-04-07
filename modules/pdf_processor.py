from pathlib import Path
from typing import List


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> List:
    """Convert a PDF into per-page RGB images (PIL.Image objects)."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    try:
        from pdf2image import convert_from_path

        return convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as pdf2image_exc:
        # Fallback for environments where Poppler/pdfinfo is not installed.
        try:
            import pypdfium2 as pdfium
        except Exception as exc:
            raise RuntimeError(
                "PDF conversion failed: pdf2image could not run (Poppler likely missing) "
                "and pypdfium2 fallback is unavailable. Install Poppler or pypdfium2."
            ) from exc

        scale = float(dpi) / 72.0
        doc = pdfium.PdfDocument(str(pdf_path))
        images = []
        for index in range(len(doc)):
            page = doc[index]
            bitmap = page.render(scale=scale)
            images.append(bitmap.to_pil())
        if not images:
            raise RuntimeError(f"No pages could be rendered from PDF: {pdf_path}")
        return images
