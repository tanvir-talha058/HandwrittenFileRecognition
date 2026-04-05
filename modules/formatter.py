import json
from io import BytesIO
from pathlib import Path
from typing import Dict


try:
    import pandas as pd
except Exception:
    pd = None

try:
    from pypdf import PdfReader, PdfWriter
except Exception:
    PdfReader = None
    PdfWriter = None

try:
    from reportlab.pdfgen import canvas
except Exception:
    canvas = None


def flatten_output(parsed: Dict) -> Dict:
    flat = {}

    for key, value in parsed.get("fields", {}).items():
        flat[key] = value

    for table_name, fields in parsed.get("tables", {}).items():
        for key, value in fields.items():
            flat[f"{table_name}.{key}"] = value

    for key, value in parsed.get("signatures", {}).items():
        flat[f"signature.{key}"] = value

    for key, value in parsed.get("stamps", {}).items():
        flat[f"stamp.{key}"] = value

    return flat


def save_json(parsed: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(parsed, handle, ensure_ascii=False, indent=2)


def save_excel(parsed: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if pd is None:
        # Write a companion JSON if pandas/openpyxl is unavailable.
        fallback = out_path.with_suffix(".json")
        save_json({"note": "Excel export fallback", "data": flatten_output(parsed)}, fallback)
        return

    flat = flatten_output(parsed)
    df = pd.DataFrame([flat])
    df.to_excel(out_path, index=False)


def save_filled_pdf_placeholder(parsed: Dict, out_path: Path):
    """Placeholder for template re-rendering; writes a simple text artifact for now."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = flatten_output(parsed)
    text_path = out_path.with_suffix(".txt")
    lines = ["Auto-filled PDF placeholder", "", "Extracted values:"]
    lines.extend([f"- {k}: {v}" for k, v in flat.items()])
    text_path.write_text("\n".join(lines), encoding="utf-8")


def save_filled_pdf(
    parsed: Dict,
    template_pdf_path: Path,
    out_pdf_path: Path,
    field_map: Dict,
    page_image_sizes: Dict[int, tuple],
):
    """Render extracted text values onto a template PDF using field bbox coordinates."""
    if PdfReader is None or PdfWriter is None or canvas is None:
        raise RuntimeError("PDF fill dependencies are missing. Install pypdf and reportlab.")

    if not template_pdf_path.exists():
        raise FileNotFoundError(f"Template PDF not found: {template_pdf_path}")

    reader = PdfReader(str(template_pdf_path))
    writer = PdfWriter()

    fields = field_map.get("fields", [])
    field_values = parsed.get("fields", {})

    fields_by_page = {}
    for field in fields:
        page_no = int(field.get("page", 1))
        fields_by_page.setdefault(page_no, []).append(field)

    for idx, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)

        packet = BytesIO()
        c = canvas.Canvas(packet, pagesize=(width, height))
        c.setFont("Helvetica", 9)

        img_w, img_h = page_image_sizes.get(idx, (1024, 1448))
        sx = width / float(img_w)
        sy = height / float(img_h)

        for field in fields_by_page.get(idx, []):
            name = field.get("name")
            value = field_values.get(name)
            if value is None:
                continue

            text = str(value).strip()
            if not text:
                continue

            x1, y1, x2, y2 = field.get("bbox", [0, 0, 0, 0])
            x_pdf = float(x1) * sx
            # Convert top-left image coordinates to PDF bottom-left coordinates.
            y_pdf = height - (float(y2) * sy) + 2.0

            max_width = max(40.0, float(x2 - x1) * sx)
            clipped = text
            if c.stringWidth(clipped, "Helvetica", 9) > max_width:
                while clipped and c.stringWidth(clipped + "...", "Helvetica", 9) > max_width:
                    clipped = clipped[:-1]
                clipped = (clipped + "...") if clipped else ""

            c.drawString(x_pdf, y_pdf, clipped)

        c.save()
        packet.seek(0)

        overlay_reader = PdfReader(packet)
        overlay_page = overlay_reader.pages[0]
        page.merge_page(overlay_page)
        writer.add_page(page)

    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf_path.open("wb") as out_handle:
        writer.write(out_handle)
